# Phase 7 on a rented NVIDIA GPU

The CUDA backend (`engine/backend_cuda.py`) and the two paged-attention ports (`engine/kernels_triton.py`,
`engine/kernels_cuda.py`) were written on the Mac and have **not run yet**. Run these steps in order. Any
GPU with bf16 support and at least 24 GB of memory works for Qwen3-1.7B (for example an L4, A10G, A100 or H100).

```bash
git clone <this repo> && cd Mac_inference_engine
uv venv && uv pip install torch triton transformers huggingface_hub numpy pytest mlx   # mlx: CPU wheel, only for test imports
huggingface-cli download Qwen/Qwen3-0.6B && huggingface-cli download Qwen/Qwen3-1.7B

# 1. correctness: paged decode (Triton + CUDA C++) vs torch SDPA; model logits vs HF transformers
uv run pytest tests/test_cuda.py -q
uv run pytest tests/test_scheduler.py -q          # backend-independent scheduler and prefix-cache stress test

# 2. serve with our kernels; --backend torch is the copy-KV-then-SDPA baseline
uv run python -m engine.server --model Qwen/Qwen3-1.7B --backend cuda --port 8000 --kv-gb 3.5 --max-seqs 16

# 3. vLLM with matched settings
uv pip install vllm
vllm serve Qwen/Qwen3-1.7B --port 8103 --dtype bfloat16 --max-model-len 8192 --max-num-seqs 16 \
  --kv-cache-memory-bytes 3758096384

# 4. same load generator and workloads as on the Mac (repeat for each port)
for W in chat long prefix; do
  uv run python bench/loadgen.py --url http://127.0.0.1:8000 --model Qwen/Qwen3-1.7B --workload $W \
    --concurrency 1,2,4,8,16 --num-requests 32 --suppress-eos --out bench/results/cuda-ours_$W.json
done

# 5. profile the decode kernel
nsys profile -o decode --trace cuda,nvtx uv run python bench/loadgen.py --url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-1.7B --workload chat --concurrency 16 --num-requests 32 --suppress-eos
ncu --set full -k regex:paged_decode -c 20 -o paged_decode uv run pytest tests/test_cuda.py -q -k "paged_decode and cuda"
```

Known gaps, in the order they matter:
- **Prefill:** prefill gathers each sequence's KV blocks and calls torch SDPA (the copy approach). A paged
  FlashAttention prefill kernel is the next port.
- **No 4-bit or speculative decoding:** both depend on Metal kernels, and the server refuses `--quant` and `--draft` on CUDA.
- **No tensor parallelism yet (the stretch goal):** it needs 2 GPUs and a `torch.distributed` all-reduce after `o` and `down`.
