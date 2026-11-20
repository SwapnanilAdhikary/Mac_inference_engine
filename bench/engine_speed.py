"""End-to-end engine speed, offline (no HTTP): decode tok/s and prefill tok/s per backend.

    uv run python bench/engine_speed.py --model Qwen/Qwen3-1.7B
mlx = copy-KV-into-a-buffer baseline (pure MLX ops), metal = our kernels."""
import argparse, sys, time
from pathlib import Path
import mlx.core as mx
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.backend import MLXBackend, MetalBackend
from engine.engine import Engine
from engine.model import Qwen3
from engine.scheduler import SamplingParams


def decode(m, bsz, ctx, steps=32):
    eng = Engine(m, bsz * (-(-(ctx + steps + 16) // 16)) + 8, 16, 8192, 64)
    reqs = [eng.add([(1000 + 7 * i + j) % 150000 for j in range(ctx)], SamplingParams(max_tokens=steps + 8, ignore_eos=True))
            for i in range(bsz)]
    while any(r.num_computed < len(r.prompt) for r in reqs):
        eng.step()                                      # prefill
    for _ in range(3):
        eng.step()
    t0 = time.perf_counter()
    for _ in range(steps):
        eng.step()
    dt = (time.perf_counter() - t0) / steps
    for r in reqs:
        eng.sched.finish(r, "abort")
    return bsz / dt, dt


def prefill(m, L, chunk=512):
    eng = Engine(m, -(-L // 16) + 8, 16, chunk, 64)
    eng.add(list(range(1000, 1000 + L)), SamplingParams(max_tokens=1, ignore_eos=True))
    eng.step(); mx.synchronize()
    eng = Engine(m, -(-L // 16) + 8, 16, chunk, 64)
    eng.add(list(range(2000, 2000 + L)), SamplingParams(max_tokens=1, ignore_eos=True))
    t0 = time.perf_counter()
    while eng.sched.has_work():
        eng.step()
    return L / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--configs", default="mlx,metal,metal-q4")
    a = ap.parse_args()
    configs = {"mlx": ("mlx (copy KV)", MLXBackend, None), "metal": ("metal (ours)", MetalBackend, None),
               "metal-q4": ("metal 4-bit", MetalBackend, 4)}
    rows = []
    for name, B, quant in [configs[c] for c in a.configs.split(",")]:
        m = Qwen3(a.model, B(), quant=quant)
        r = {"backend": name}
        for bsz, ctx in [(1, 128), (1, 4096), (16, 128), (16, 2048), (64, 512)]:
            r[f"decode b={bsz} ctx={ctx}"] = decode(m, bsz, ctx)[0]
        for L in (512, 4096):
            r[f"prefill {L}"] = prefill(m, L)
        rows.append(r)
        del m; mx.clear_cache()
    keys = [k for k in rows[0] if k != "backend"]
    print(f"{'tok/s':24s}" + "".join(f"{r['backend']:>16s}" for r in rows) + "   (ratio vs first column)")
    for k in keys:
        print(f"{k:24s}" + "".join(f"{r[k]:16.1f}" for r in rows) + "   " +
              " ".join(f"{r[k] / rows[0][k]:.2f}x" for r in rows[1:]))


if __name__ == "__main__":
    main()
