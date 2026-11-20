# bench: baselines

- `loadgen.py`: closed-loop load generator (stdlib asyncio HTTP and SSE). For each concurrency level it reports TTFT, ITL (gaps between SSE chunks), TPOT, E2E, output tok/s and req/s, and it checks every request's output length against `max_tokens`.
- `bandwidth.py`: measures MLX GPU memory bandwidth and prints the batch-1 decode ceiling for Qwen3-0.6B and Qwen3-1.7B.

Run only one engine at a time. Run real sweeps on an idle GPU: other GPU work shifts the numbers by 10% or more. To check the GPU, run `ioreg -r -d 1 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*'`.

## Bandwidth and decode ceiling (Apple M5, 16 GB, 2026-11-20, idle GPU)

```
$ HF_HUB_OFFLINE=1 uv run python bench/bandwidth.py
  copy (x+1, read+write)     133.2 GB/s  (86.7% of 153.6 GB/s spec)
  reduce (sum, read)         133.8 GB/s  (87.1% of 153.6 GB/s spec)
  gemv bf16 (W@v, read)      125.3 GB/s  (81.6% of 153.6 GB/s spec)
read-only (decode-like) bandwidth: 133.8 GB/s
  Qwen/Qwen3-0.6B: files 1.503 GB, unique weights 1.192 GB ->  112.2 tok/s @ measured,  128.8 tok/s @ spec
  Qwen/Qwen3-1.7B: files 4.064 GB, unique weights 3.441 GB ->   38.9 tok/s @ measured,   44.6 tok/s @ spec
```

These checkpoints store `lm_head.weight` as a full copy of `embed_tokens` even though `tie_word_embeddings=true`. A tied engine reads that matrix once per token, so the ceiling uses unique bytes. If the duplicate is loaded and used as a separate lm_head, each token reads an extra 0.31 GB (0.6B) or 0.62 GB (1.7B), and the ceiling drops to 89 tok/s and 33 tok/s. Measured bandwidth was 120 to 136 GB/s depending on load, so expect run-to-run noise of a few percent.

## Matched settings

| | llama.cpp | mlx-lm | vllm-metal |
|---|---|---|---|
| Version | brew `llama.cpp` 1.0.0 (b12359-b1b770d90) | mlx-lm 0.32.2, mlx 0.32.4 (project env) | vllm 0.34.0+cpu, vllm-metal 0.34.0.dev20261119200334 (`~/.venv-vllm-metal`) |
| Weights | `unsloth/Qwen3-*-GGUF` `*-BF16.gguf` | `Qwen/Qwen3-*` safetensors (bf16) | `Qwen/Qwen3-*` safetensors, `--dtype bfloat16` |
| Max context per request | 8192 (`--kv-unified-per-slot 8192`) | no setting (model max, 40960) | 8192 (`--max-model-len`) |
| KV pool | 32768 tokens (`-c 32768 -kvu`) | no limit; allocated on demand | 32768 tokens (`--kv-cache-memory-bytes 3758096384`) |
| Max concurrent sequences | 16 (`-np 16`) | 16 (`--decode-concurrency 16`) | 16 (`--max-num-seqs 16`) |
| Prefix caching (default) | ON: `--cache-prompt` (reuses the slot's previous prompt) plus host RAM cache `--cache-ram` (default 8 GiB, capped here at 2 GiB). Concurrent requests do not share a prefix that is still being computed. | ON: LRU prompt cache of 10 finished-request KV caches | ON: `--enable-prefix-caching` (block-level and shared) |
| Prefix caching OFF | `--no-cache-prompt --cache-ram 0` | `--prompt-cache-size 0` (from the code; not run) | `--disable-prefix-caching` |
| `ignore_eos` | honored (EOS logits set to -inf) | **ignored**: use `loadgen --suppress-eos` | honored (EOS is sampled, then generation continues) |

The KV budget is 114,688 B/token for both models (28 layers × 8 KV heads × 128 dim × K+V × 2 B), so 32768 tokens = 3.76 GB. At concurrency 16, the `long` workload (16 × about 4.2k tokens) and the unshared `prefix` workload need more KV than this pool. llama.cpp and vLLM will then queue or preempt, but mlx-lm will just allocate more. You can cap those sweeps at concurrency 8, or keep the queueing as a measured result.

Use `--suppress-eos` (sends `logit_bias` of -100 on token ids 151643 and 151645) on **every** engine. That way all engines get byte-identical requests and generate identical greedy token streams. With `ignore_eos` alone, vLLM generates the EOS token and keeps going, while llama.cpp never samples EOS, so the generated text differs between them. The token counts match either way.

## Start servers (replace 0.6B with 1.7B for the main model)

```bash
# llama.cpp  (port 8101)
GGUF=$(ls ~/.cache/huggingface/hub/models--unsloth--Qwen3-0.6B-GGUF/snapshots/*/Qwen3-0.6B-BF16.gguf)
llama-server -m $GGUF --host 127.0.0.1 --port 8101 -ngl 99 -fa on \
  -c 32768 -np 16 -kvu --kv-unified-per-slot 8192 --cache-ram 2048

# mlx-lm  (port 8102); the project env
HF_HUB_OFFLINE=1 uv run mlx_lm.server --model Qwen/Qwen3-0.6B --host 127.0.0.1 --port 8102 \
  --decode-concurrency 16

# vllm-metal  (port 8103); separate venv
HF_HUB_OFFLINE=1 ~/.venv-vllm-metal/bin/vllm serve Qwen/Qwen3-0.6B --host 127.0.0.1 --port 8103 \
  --dtype bfloat16 --max-model-len 8192 --kv-cache-memory-bytes 3758096384 --max-num-seqs 16
```

Readiness checks: `curl -s :8101/health`, `curl -s :8102/v1/models`, `curl -s :8103/v1/models`.

At startup, vllm-metal runs a "KV commit probe". If macOS pages memory out while the probe allocates the explicit `--kv-cache-memory-bytes`, the server refuses to start (this happened at 4.7 GB with about 5 GB free). Free some memory, lower the budget, or set `VLLM_METAL_KV_COMMIT_PROBE=0` to skip the probe. The pool must also hold one block more than `--max-model-len`, so a 16384-token pool rejects `--max-model-len 16384`. The vllm-metal venv has its own mlx-lm (0.32.3.dev), so it does not use the project's 0.32.2.

## Run loadgen

```bash
for W in chat long prefix; do
  HF_HUB_OFFLINE=1 uv run python bench/loadgen.py --url http://127.0.0.1:8101 --model Qwen/Qwen3-1.7B \
    --workload $W --concurrency 1,2,4,8,16 --num-requests 32 --suppress-eos \
    --out bench/results/llamacpp_1.7b_$W.json
done
# smoke test: --concurrency 1,2 --num-requests 4 --workload chat --max-tokens 32
```

- `--model` must be the served model name. On mlx-lm, a different name makes the server try to **load that model**. On vLLM, a different name returns 404. llama.cpp ignores the field.
- Default output lengths: `chat` draws 128 to 256 per request from the seed, `long` and `prefix` use 64. `--max-tokens N` sets a fixed length for every request.
- `--endpoint chat` uses `/v1/chat/completions`. For `prefix`, the shared prefix becomes the system message. Reasoning text counts as output: llama.cpp sends it in `reasoning_content`, while vLLM leaves `<think>` inside `content`.
- Every level gets new prompts, and `prefix` gets a new shared prefix per level, so a prefix cache warmed by one level cannot help the next. Requests are seeded, so they are identical across engines.
- Output token counts come from the server's `usage.completion_tokens` (all three engines report it when sent `stream_options.include_usage`). The JSON also stores the count from re-tokenizing the text. `len_bad` counts requests with output length ≠ `max_tokens`. If it is non-zero, the engine ignored `ignore_eos`.
- ITL is measured between SSE chunks. All three engines sent 1 token per chunk in the smoke tests (`chunks_per_request_mean` = `output_tokens_mean`). If an engine starts sending several tokens per chunk, use TPOT = (E2E − TTFT) / (n − 1) instead.
- Prompt token counts from our HF tokenizer exactly matched the server-reported `prompt_tokens` on all three engines.

## Smoke test (Qwen3-0.6B, chat, 4 requests per level, max_tokens 32; GPU shared with other work)

| engine | conc | out tok/s | TTFT p50 ms | ITL p50 / p99 ms | TPOT p50 ms | len_bad |
|---|---|---|---|---|---|---|
| llama.cpp | 1 | 88.5 | 45.7 | 10.19 / 11.01 | 10.24 | 0 |
| llama.cpp | 2 | 150.8 | 60.8 | 11.66 / 13.68 | 11.73 | 0 |
| mlx-lm (`--suppress-eos`) | 1 | 80.6 | 58.6 | 10.96 / 11.41 | 10.99 | 0 |
| mlx-lm (`--suppress-eos`) | 2 | 136.8 | 92.2 | 11.03 / 22.35 | 11.65 | 0 |
| vllm-metal | 1 | 67.4 | 48.3 | 12.30 / 61.19 | 12.17 | 0 |
| vllm-metal | 2 | 115.7 | 60.4 | 11.34 / 227.43 | 15.83 | 0 |

The vllm-metal smoke test used `--max-model-len 8192 --kv-cache-memory-bytes 1879048192`. The 4.7 GB budget was refused because of memory pressure. With only 4 requests per level, these numbers show that the setup works. They are not benchmark results.
