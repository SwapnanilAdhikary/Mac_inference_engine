#!/usr/bin/env bash
# Phase 6 sweep: every engine, same model + matched settings (see bench/README.md), one engine at a time.
#   bash bench/sweep.sh [config ...]      configs: ours llamacpp mlxlm vllm ours-noprefix ours-q4 ours-spec
# Results: bench/results/<config>_<workload>.json
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
MODEL=Qwen/Qwen3-1.7B
CONC=${CONC:-1,2,4,8,16}
NREQ=${NREQ:-32}
WORKLOADS=${WORKLOADS:-chat long prefix}
OUT=bench/results
mkdir -p "$OUT"
GGUF=$(ls ~/.cache/huggingface/hub/models--unsloth--Qwen3-1.7B-GGUF/snapshots/*/Qwen3-1.7B-BF16.gguf)
OURS="uv run python -m engine.server --model $MODEL --kv-gb 3.5 --max-seqs 16"

start() {  # name port cmd...
  local name=$1 port=$2; shift 2
  echo "=== $name: starting on :$port"
  "$@" > "$OUT/$name.server.log" 2>&1 &
  PID=$!
  for _ in $(seq 1 600); do
    curl -sf "http://127.0.0.1:$port/v1/models" > /dev/null && return 0
    kill -0 $PID 2>/dev/null || { echo "$name died; see $OUT/$name.server.log"; return 1; }
    sleep 1
  done
  echo "$name not ready after 600s"; return 1
}

stop() {
  pkill -P $PID 2>/dev/null; kill $PID 2>/dev/null; wait $PID 2>/dev/null
  sleep 5   # let the GPU memory go back before the next engine
}

load() {  # name port [workloads]
  local name=$1 port=$2 wl=${3:-$WORKLOADS}
  for W in $wl; do
    echo "--- $name / $W"
    uv run python bench/loadgen.py --url "http://127.0.0.1:$port" --model "$MODEL" --workload "$W" \
      --concurrency "$CONC" --num-requests "$NREQ" --suppress-eos --out "$OUT/${name}_$W.json" 2>&1 | grep -v PyTorch
  done
}

run() {
  case $1 in
    ours)          start ours 8000 $OURS --port 8000 && load ours 8000 ;;
    ours-noprefix) start ours-noprefix 8000 $OURS --port 8000 --no-prefix-caching && load ours-noprefix 8000 "chat prefix" ;;
    ours-q4)       start ours-q4 8000 $OURS --port 8000 --quant 4 && load ours-q4 8000 ;;
    ours-spec)     start ours-spec 8000 $OURS --port 8000 --draft Qwen/Qwen3-0.6B --spec-k 4 && load ours-spec 8000 chat ;;
    llamacpp)      start llamacpp 8101 llama-server -m "$GGUF" --host 127.0.0.1 --port 8101 -ngl 99 -fa on \
                     -c 32768 -np 16 -kvu --kv-unified-per-slot 8192 --cache-ram 2048 && load llamacpp 8101 ;;
    mlxlm)         start mlxlm 8102 uv run mlx_lm.server --model $MODEL --host 127.0.0.1 --port 8102 \
                     --decode-concurrency 16 && load mlxlm 8102 ;;
    vllm)          start vllm 8103 ~/.venv-vllm-metal/bin/vllm serve $MODEL --host 127.0.0.1 --port 8103 \
                     --dtype bfloat16 --max-model-len 8192 --kv-cache-memory-bytes 3758096384 --max-num-seqs 16 \
                     && load vllm 8103 ;;
    *) echo "unknown config $1"; return 1 ;;
  esac
  stop
}

for c in "${@:-ours llamacpp mlxlm vllm ours-noprefix ours-q4 ours-spec}"; do
  for cc in $c; do run "$cc"; done
done
