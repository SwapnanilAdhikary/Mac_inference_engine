"""Measure achievable GPU memory bandwidth with MLX and derive the decode ceiling.

Decode at batch 1 is memory-bound: every generated token must stream all weights
from DRAM once, so  max tokens/s ~= bandwidth / weight_bytes  (ignores KV-cache
reads and activations, so it is an upper bound).

Qwen3-0.6B/1.7B checkpoints store `lm_head.weight` as a separate copy of
`model.embed_tokens.weight` even though `tie_word_embeddings` is true. A tied
engine reads that matrix once per token (as lm_head; the embedding lookup only
touches one row), so the ceiling uses the de-duplicated byte count.

    HF_HUB_OFFLINE=1 uv run python bench/bandwidth.py [--mib 1024] [--bufs 4] [--iters 20]
"""

import argparse
import glob
import json
import os
import struct
import time

import mlx.core as mx
from huggingface_hub import snapshot_download

M5_SPEC_GBPS = 153.6  # Apple spec for base M5 unified memory
MODELS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"]


def best_time(fn, iters, warmup=5):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def measure(mib, bufs, iters):
    """Several independent buffers per eval amortize the per-eval sync overhead."""
    n = mib * 2**20 // 2 // bufs  # bf16 elements per buffer
    k = 4096
    xs = [mx.random.normal((n,)).astype(mx.bfloat16) for _ in range(bufs)]
    ws = [x.reshape(n // k, k) for x in xs]  # weight-like matrices for GEMV
    v = mx.random.normal((k, 1)).astype(mx.bfloat16)
    mx.eval(xs, v)
    nbytes = bufs * n * 2
    tests = {  # name: (fn, bytes moved)
        "copy (x+1, read+write)": (lambda: [x + 1 for x in xs], 2 * nbytes),
        "reduce (sum, read)": (lambda: [mx.sum(x) for x in xs], nbytes),
        "gemv bf16 (W@v, read)": (lambda: [w @ v for w in ws], nbytes),
    }
    return {name: moved / best_time(fn, iters) / 1e9 for name, (fn, moved) in tests.items()}


def weight_bytes(repo):
    """(file bytes, unique tensor bytes) from the safetensors in the HF cache, or None."""
    try:
        path = snapshot_download(repo, local_files_only=True, allow_patterns=["*.safetensors", "*.json"])
    except Exception:
        return None
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index):  # sharded: make sure every shard is present
        with open(index) as f:
            files = [os.path.join(path, s) for s in set(json.load(f)["weight_map"].values())]
        if not all(os.path.exists(p) for p in files):
            return None
    else:
        files = glob.glob(os.path.join(path, "*.safetensors"))
    if not files:
        return None
    sizes = {}
    for fn in files:
        with open(fn, "rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        sizes.update({k: v["data_offsets"][1] - v["data_offsets"][0] for k, v in header.items() if k != "__metadata__"})
    with open(os.path.join(path, "config.json")) as f:
        tied = json.load(f).get("tie_word_embeddings", False)
    unique = sum(sizes.values()) - (sizes.get("lm_head.weight", 0) if tied else 0)
    return sum(os.path.getsize(f) for f in files), unique


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mib", type=int, default=1024, help="total buffer size in MiB (>> SLC)")
    ap.add_argument("--bufs", type=int, default=4, help="independent buffers evaluated together")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    print(f"device: {mx.device_info()['device_name']}  {args.bufs} x {args.mib // args.bufs} MiB bf16  best of {args.iters}")
    bw = measure(args.mib, args.bufs, args.iters)
    for name, gbps in bw.items():
        print(f"  {name:24s} {gbps:7.1f} GB/s  ({100 * gbps / M5_SPEC_GBPS:4.1f}% of {M5_SPEC_GBPS} GB/s spec)")
    read_bw = max(bw["reduce (sum, read)"], bw["gemv bf16 (W@v, read)"])
    print(f"read-only (decode-like) bandwidth: {read_bw:.1f} GB/s")

    print("\ndecode ceiling at batch 1 = bandwidth / bytes read per token")
    for repo in MODELS:
        wb = weight_bytes(repo)
        if wb is None:
            print(f"  {repo}: safetensors not (fully) in HF cache")
            continue
        files, unique = wb
        print(f"  {repo}: files {files / 1e9:.3f} GB, unique weights {unique / 1e9:.3f} GB -> "
              f"{read_bw * 1e9 / unique:6.1f} tok/s @ measured, {M5_SPEC_GBPS * 1e9 / unique:6.1f} tok/s @ spec "
              f"(file-size based: {read_bw * 1e9 / files:.1f} / {M5_SPEC_GBPS * 1e9 / files:.1f})")


if __name__ == "__main__":
    main()
