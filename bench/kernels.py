"""Kernel benchmarks: achieved GB/s per kernel, and paged attention vs the copy-KV-then-attend baseline.

    uv run python bench/kernels.py
Shapes are Qwen3-1.7B's (H=16, HKV=8, D=128, hidden=2048, intermediate=6144)."""
import itertools, sys, time
from pathlib import Path
import mlx.core as mx
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import kernels as K
from engine.backend import MLXBackend
from engine.model import Batch, SeqSlice

H, HKV, D, HID, I, BS = 16, 8, 128, 2048, 6144, 16
BW_PEAK = 133.8  # GB/s measured by bench/bandwidth.py
ref = MLXBackend()


def bench(fn, reps=20, inner=10):
    """Median seconds per call. `inner` calls are queued per eval so launch overhead is amortized
    the way it is inside a real forward pass."""
    for _ in range(3):
        mx.eval([fn() for _ in range(inner)])
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval([fn() for _ in range(inner)])
        ts.append((time.perf_counter() - t0) / inner)
    return float(np.median(ts))


def row(name, t, nbytes):
    gbs = nbytes / t / 1e9
    print(f"  {name:44s} {t * 1e6:9.1f} us  {gbs:7.1f} GB/s  ({100 * gbs / BW_PEAK:4.0f}% of measured peak)")


def b16(*shape):
    x = mx.random.normal(shape).astype(mx.bfloat16)
    mx.eval(x)
    return x


def main():
    print("RMSNorm (fused residual add) / SiLU-mul / QK-norm+RoPE+KV-write (R=16 is launch-bound; "
          "R=512 working sets fit in the SLC, so GB/s can exceed DRAM peak)")
    for R in (16, 512):
        x, d, w = b16(R, HID), b16(R, HID), b16(HID)
        row(f"add_rms_norm R={R}", bench(lambda: K.rms_norm(x, w, 1e-6, delta=d)), 5 * R * HID * 2)
        row(f"  ref (mlx ops) R={R}", bench(lambda: ref.add_rms_norm(x, d, w, 1e-6)), 5 * R * HID * 2)
        gu = b16(R, 2 * I)
        row(f"silu_mul R={R}", bench(lambda: K.silu_mul(gu)), 3 * R * I * 2)
        row(f"  ref (mlx ops) R={R}", bench(lambda: ref.silu_mul(gu)), 3 * R * I * 2)
        qkv = b16(R, (H + 2 * HKV) * D)
        kc, vc = b16(65536, HKV, D), b16(65536, HKV, D)
        inv = 1e6 ** (-mx.arange(0, D, 2, dtype=mx.float32) / D)
        qn, kn = b16(D), b16(D)
        pos = mx.arange(R, dtype=mx.int32) + 100
        slots = mx.array(np.random.default_rng(0).permutation(65536)[:R].astype(np.int32))
        nb = 2 * R * (H + 2 * HKV) * D * 2 // 2 * 2  # read qkv + write q,k,v
        row(f"qk_norm_rope_write T={R}", bench(lambda: K.qk_norm_rope_write(qkv, qn, kn, inv, pos, slots, kc, vc,
                                                                           H, HKV, 1e-6)[0]), nb)

    print("\nPaged decode attention (bytes = K+V actually attended; KV pools rotated past the SLC)")
    rng = np.random.default_rng(0)
    for S, ctx in [(1, 1024), (1, 8192), (4, 8192), (16, 512), (16, 2048), (16, 4096), (64, 1024)]:
        nblk = -(-ctx // BS)
        nb = S * ctx * HKV * D * 2 * 2
        R = max(1, -(-512 * 2**20 // nb))                    # rotate over >= 512 MiB of KV
        kc, vc = b16(R * S * nblk * BS, HKV, D), b16(R * S * nblk * BS, HKV, D)
        perm = rng.permutation(R * S * nblk).astype(np.int32)
        bts = [perm[r * S * nblk:(r + 1) * S * nblk].reshape(S, nblk) for r in range(R)]
        q = b16(S, H, D)
        rows, cl = mx.arange(S, dtype=mx.int32), mx.array(np.full(S, ctx, np.int32))
        dep = mx.zeros((1,), mx.int32)
        btm = [mx.array(b) for b in bts]
        it = itertools.cycle(range(R))
        t1 = bench(lambda: K.paged_decode(q, rows, kc, vc, btm[next(it)], cl, dep, H, HKV, BS, ctx))
        row(f"S={S:3d} ctx={ctx:5d} paged", t1, nb)
        t2 = bench(lambda: K.paged_decode(q, rows, kc, vc, btm[next(it)], cl, dep, H, HKV, BS, ctx, part=512))
        row(f"S={S:3d} ctx={ctx:5d} paged split(512)", t2, nb)
        metas = [ref.prepare_attention(Batch([0] * S, [ctx - 1] * S, [0] * S,
                                             [SeqSlice(i, 1, ctx, list(b[i])) for i in range(S)], []), BS)
                 for b in bts]
        t3 = bench(lambda: ref.attention(q, [kc, vc], metas[next(it)], D ** -0.5))
        row(f"S={S:3d} ctx={ctx:5d} copy-KV baseline", t3, nb)
        print(f"  {'':44s} paged speedup vs copy: {t3 / min(t1, t2):.2f}x")
        del kc, vc

    print("\nPaged prefill (flash, causal). TFLOP/s counts only the causal half; "
          "mx.fast.sdpa on dense K/V is an upper-bound reference")
    for L, prev in [(512, 0), (512, 2048), (512, 8192), (2048, 0)]:
        ctx = prev + L
        nblk = -(-ctx // BS)
        kc, vc = b16(nblk * BS, HKV, D), b16(nblk * BS, HKV, D)
        bt = rng.permutation(nblk)[None].astype(np.int32)
        q = b16(L, H, D)
        tiles = [[0, j, min(K.BQ, L - j), prev + j] for j in range(0, L, K.BQ)]
        tl, btm, cl = mx.array(np.array(tiles, np.int32)), mx.array(bt), mx.array([ctx], dtype=mx.int32)
        dep = mx.zeros((1,), mx.int32)
        flops = 4 * H * D * L * (prev + L / 2)
        res = {}
        for nax in ([True] if K.has_nax() else []) + [False]:
            res["nax" if nax else "simdgroup"] = bench(lambda: K.paged_prefill(
                q, tl, len(tiles), kc, vc, btm, cl, dep, H, HKV, BS, L, nax=nax), reps=10, inner=3)
        batch = Batch([0] * L, list(range(prev, ctx)), [0] * L, [SeqSlice(0, L, ctx, list(bt[0]))], [])
        meta = ref.prepare_attention(batch, BS)
        res["copy-KV baseline"] = bench(lambda: ref.attention(q, [kc, vc], meta, D ** -0.5), reps=10, inner=3)
        qd, kd, vd = q.transpose(1, 0, 2)[None], b16(1, HKV, ctx, D), b16(1, HKV, ctx, D)
        res["mx.fast.sdpa (ref)"] = bench(lambda: mx.fast.scaled_dot_product_attention(
            qd, kd, vd, scale=D ** -0.5, mask="causal"), reps=10, inner=3)
        print(f"  L={L} cached={prev}: " + "  ".join(
            f"{k} {t * 1e3:.2f}ms/{flops / t / 1e12:.1f}TF" for k, t in res.items()))
    bench_q4()


def bench_q4():
    print("\n4-bit matmul (x[M,K] @ W[N,K]^T, W = gate|up of Qwen3-1.7B: N=12288, K=2048; bytes = weights read)")
    N, Kd = 2 * I, HID
    ws = [b16(N, Kd) for _ in range(4)]                       # rotate 4 weights (200 MB bf16) past the SLC
    qs = [K.quantize4(w) for w in ws]
    mx.eval(qs)
    qbytes = N * Kd // 2 + 2 * N * (Kd // K.GROUP) * 2
    for M in (1, 4, 8, 16, 64, 512):
        x = b16(M, Kd)
        it = itertools.cycle(range(4))
        tv = bench(lambda: K.qmv(x, *qs[next(it)])) if M <= 8 else float("nan")
        tm = bench(lambda: K.qmm(x, *qs[next(it)])) if K.has_nax() else float("nan")
        td = bench(lambda: x @ ws[next(it)].T)
        tdq = bench(lambda: x @ K.dequantize4(*qs[next(it)]).T) if M >= 64 else float("nan")
        tq = min(t for t in (tv, tm, tdq) if t == t)
        print(f"  M={M:4d}  qmv {tv * 1e6:7.1f} us  qmm(nax) {tm * 1e6:7.1f} us  dequant+gemm {tdq * 1e6:7.1f} us"
              f"  -> best {qbytes / tq / 1e9:5.1f} GB/s   bf16 {td * 1e6:7.1f} us ({N * Kd * 2 / td / 1e9:5.1f} GB/s)"
              f"   q4 speedup {td / tq:5.2f}x")
    p, s_, b_ = qs[0]
    t = bench(lambda: K.dequantize4(p, s_, b_))
    row("dequantize4 (read 4-bit, write bf16)", t, qbytes + N * Kd * 2)


if __name__ == "__main__":
    main() if "--q4" not in sys.argv else bench_q4()
