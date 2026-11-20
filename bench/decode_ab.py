"""End-to-end decode-step A/B for matmul routing (the benchmark the bmm routing rule was decided with).

    uv run python bench/decode_ab.py

Why not a kernel microbenchmark: bench/kernels.py queues independent calls per eval, so the GPU overlaps them
and it measures throughput. Inside a forward pass every matmul waits for the previous op, so what matters is
single-launch latency, and kernels with few threadgroups that looked 12% faster there were 20% slower here."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench.engine_speed as E
from engine import kernels as K
from engine.backend import MetalBackend
from engine.model import Qwen3


def main():
    m = Qwen3("Qwen/Qwen3-1.7B", MetalBackend())
    orig = K.use_bmm
    configs = {"MLX GEMM only": lambda M, N: False, "bmm on gate|up + lm_head (engine default)": orig}
    for b in (8, 12, 14, 16):
        res = {}
        for _ in range(3):                                   # interleaved trials; report the best of each
            for name, rule in configs.items():
                K.use_bmm = rule
                res.setdefault(name, []).append(E.decode(m, b, 400, steps=40)[1] * 1e3)
        K.use_bmm = orig
        print(f"batch {b:2d}: " + "   ".join(f"{k} {min(v):5.1f} ms/step" for k, v in res.items()))


if __name__ == "__main__":
    main()
