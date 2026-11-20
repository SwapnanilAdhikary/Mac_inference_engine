"""Each Metal kernel vs a reference op (mx.fast.* allowed here, as reference only)."""
import mlx.core as mx
import numpy as np
import pytest
from engine import kernels as K
from engine.backend import MLXBackend

BS, D = 16, 128
ref = MLXBackend()
rng = np.random.default_rng(0)


def close(a, b, tol, rtol=0.02):
    """|a - b| <= tol + rtol*|b| elementwise (rtol ~ a few bf16 ulps)."""
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    err = (mx.abs(a - b) - rtol * mx.abs(b)).max().item()
    assert err <= tol, err


@pytest.mark.parametrize("R,N", [(1, 1024), (7, 2048), (300, 2048)])
def test_rms_norm(R, N):
    x = mx.random.normal((R, N)).astype(mx.bfloat16) * 3
    d = mx.random.normal((R, N)).astype(mx.bfloat16)
    w = (1 + 0.1 * mx.random.normal((N,))).astype(mx.bfloat16)
    close(K.rms_norm(x, w, 1e-6), mx.fast.rms_norm(x, w, 1e-6), 0.07)
    close(K.rms_norm(x, w, 1e-6), ref.rms_norm(x, w, 1e-6), 0.07)
    out, xo = K.rms_norm(x, w, 1e-6, delta=d)
    assert mx.array_equal(xo, x + d).item()
    close(out, ref.rms_norm(x + d, w, 1e-6), 0.07)


def test_silu_mul():
    gu = mx.random.normal((9, 2 * 3072)).astype(mx.bfloat16) * 2
    close(K.silu_mul(gu), ref.silu_mul(gu), 0.06)


def test_rope_matches_mx_fast():
    x = mx.random.normal((50, 16, D)).astype(mx.bfloat16)
    for off in (0, 17, 30000):
        ours = ref.rope(x, mx.arange(off, off + 50, dtype=mx.int32), 1e6)
        theirs = mx.fast.rope(x.transpose(1, 0, 2)[None], D, traditional=False, base=1e6, scale=1.0, offset=off)
        close(ours, theirs[0].transpose(1, 0, 2), 0.02)


@pytest.mark.parametrize("H,HKV,T", [(16, 8, 1), (16, 8, 37), (32, 8, 5)])
def test_qk_norm_rope_write(H, HKV, T):
    qkv = mx.random.normal((T, (H + 2 * HKV) * D)).astype(mx.bfloat16)
    qn, kn = [(1 + 0.2 * mx.random.normal((D,))).astype(mx.bfloat16) for _ in range(2)]
    pos = mx.array(rng.integers(0, 40000, T).astype(np.int32))
    slots = mx.array(rng.permutation(4096)[:T].astype(np.int32))
    kc, vc = mx.zeros((4096, HKV, D), mx.bfloat16), mx.zeros((4096, HKV, D), mx.bfloat16)
    mx.eval(kc, vc)
    inv = 1e6 ** (-mx.arange(0, D, 2, dtype=mx.float32) / D)
    q, dep = K.qk_norm_rope_write(qkv, qn, kn, inv, pos, slots, kc, vc, H, HKV, 1e-6)
    mx.eval(q, dep)
    rq = ref.rope(ref.rms_norm(qkv[:, :H * D].reshape(T, H, D), qn, 1e-6), pos, 1e6)
    rk = ref.rope(ref.rms_norm(qkv[:, H * D:(H + HKV) * D].reshape(T, HKV, D), kn, 1e-6), pos, 1e6)
    close(q, rq, 0.07)
    close(kc[slots], rk, 0.07)
    assert mx.array_equal(vc[slots], qkv[:, (H + HKV) * D:].reshape(T, HKV, D)).item()
    untouched = np.setdiff1d(np.arange(4096), np.array(slots))
    assert mx.abs(kc[mx.array(untouched)]).max().item() == 0


def make_cache(ctx_lens, HKV, extra_blocks=8):
    nb = sum(-(-c // BS) for c in ctx_lens) + extra_blocks
    perm = rng.permutation(nb)
    tables, i = [], 0
    for c in ctx_lens:
        n = -(-c // BS)
        tables.append(perm[i:i + n]); i += n
    maxb = max(len(t) for t in tables)
    bt = np.zeros((len(ctx_lens), maxb), np.int32)
    for s, t in enumerate(tables):
        bt[s, :len(t)] = t
    kc = mx.random.normal((nb * BS, HKV, D)).astype(mx.bfloat16)
    vc = mx.random.normal((nb * BS, HKV, D)).astype(mx.bfloat16)
    return kc, vc, bt


def dense_ref(q, kc, vc, table, ctx, qpos, H, HKV):
    """q [Lq, H, D] at positions qpos -> causal attention over the first ctx cached keys."""
    j = np.arange(ctx)
    sl = mx.array((table[j // BS] * BS + j % BS).astype(np.int32))
    k, v = kc[sl], vc[sl]
    mask = mx.array(np.where(j[None, :] <= np.asarray(qpos)[:, None], 0, -np.inf).astype(np.float32))
    o = mx.fast.scaled_dot_product_attention(q.transpose(1, 0, 2)[None], k.transpose(1, 0, 2)[None],
                                             v.transpose(1, 0, 2)[None], scale=D ** -0.5, mask=mask.astype(q.dtype))
    return o[0].transpose(1, 0, 2)


@pytest.mark.parametrize("H,HKV", [(16, 8), (32, 8)])
@pytest.mark.parametrize("part", [None, 64, 256])
def test_paged_decode(H, HKV, part):
    ctx_lens = [1, BS - 1, BS, BS + 1, 300, 1000, 4097]
    kc, vc, bt = make_cache(ctx_lens, HKV)
    S = len(ctx_lens)
    q = mx.random.normal((S + 3, H, D)).astype(mx.bfloat16)
    rows = np.array([3, 0, 5, 1, 9, 2, 4], np.int32)  # arbitrary query rows
    dep = mx.zeros((1,), mx.int32)
    out = K.paged_decode(q, mx.array(rows), kc, vc, mx.array(bt), mx.array(np.array(ctx_lens, np.int32)), dep,
                         H, HKV, BS, max(ctx_lens), part=part)
    for s, c in enumerate(ctx_lens):
        r = dense_ref(q[int(rows[s]):int(rows[s]) + 1], kc, vc, bt[s], c, [c - 1], H, HKV)
        close(out[s:s + 1], r, 0.02)


def build_tiles(chunks):
    """chunks: [(seq, first_row, n_rows, first_pos)] -> tile rows of at most BQ."""
    t = []
    for seq, r0, n, p0 in chunks:
        for i in range(0, n, K.BQ):
            t.append([seq, r0 + i, min(K.BQ, n - i), p0 + i])
    return mx.array(np.array(t, np.int32)), len(t)


@pytest.mark.parametrize("nax", [False] + ([True] if K.has_nax() else []))
@pytest.mark.parametrize("H,HKV", [(16, 8), (32, 8)])
def test_paged_prefill(H, HKV, nax):
    # (cached prefix length, new chunk length): block boundaries, tile boundaries, chunked continuation
    specs = [(0, 1), (0, BS - 1), (0, BS), (0, BS + 1), (0, 100), (37, 64), (500, 33), (1, 1), (2000, 130)]
    ctx_lens = [p + n for p, n in specs]
    kc, vc, bt = make_cache(ctx_lens, HKV)
    T = sum(n for _, n in specs)
    q = mx.random.normal((T, H, D)).astype(mx.bfloat16)
    chunks, r = [], 0
    for s, (p, n) in enumerate(specs):
        chunks.append((s, r, n, p)); r += n
    tiles, nt = build_tiles(chunks)
    out = K.paged_prefill(q, tiles, nt, kc, vc, mx.array(bt), mx.array(np.array(ctx_lens, np.int32)),
                          mx.zeros((1,), mx.int32), H, HKV, BS, T, nax=nax)
    for s, (p, n) in enumerate(specs):
        r0 = chunks[s][1]
        ref_o = dense_ref(q[r0:r0 + n], kc, vc, bt[s], p + n, list(range(p, p + n)), H, HKV)
        close(out[r0:r0 + n], ref_o, 0.02)


@pytest.mark.parametrize("M", [1, 2, 5, 8, 9, 33])
@pytest.mark.parametrize("N,Kd", [(1024, 2048), (3, 256), (12288, 2048), (2048, 6144)])
def test_qlinear(M, N, Kd):
    zeros = mx.random.uniform(shape=(N, Kd)) < 0.1
    w = mx.where(zeros, 0, mx.random.normal((N, Kd)) * 0.05).astype(mx.bfloat16)
    x = mx.random.normal((M, Kd)).astype(mx.bfloat16)
    p, s, b = K.quantize4(w)
    wd = K.dequantize4(p, s, b).astype(mx.float32)
    step = mx.repeat(mx.abs(s.astype(mx.float32)), K.GROUP, axis=1)
    # error <= half a step inside the grid; up to ~1 step at the far edge that the zero-exact snap clips
    assert (mx.abs(wd - w.astype(mx.float32)) <= 1.1 * step + 1e-3).all().item()
    # zero-exact: exact zeros stay (bf16-rounding-)close to zero, not up to half a step away
    assert (mx.where(zeros, mx.abs(wd), 0) <= 0.1 * step).all().item()
    # reference dequant with MLX's own quantization format as a cross-check of our layout
    rq = mx.dequantize(p, s, b, group_size=K.GROUP, bits=4)
    close(wd, rq, 1e-3)
    ref_out = x.astype(mx.float32) @ wd.astype(mx.float32).T
    close(K.qlinear(x, p, s, b), ref_out, 0.05, rtol=0.02)
    if M <= 8:
        close(K.qmv(x, p, s, b), ref_out, 0.05, rtol=0.02)
    if K.has_nax():
        close(K.qmm(x, p, s, b), ref_out, 0.05, rtol=0.02)
