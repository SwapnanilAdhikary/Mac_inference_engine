"""Turn bench/results/<config>_<workload>.json into markdown tables (printed; paste into RESULTS.md).

    uv run python bench/report.py [config ...]"""
import json, sys
from pathlib import Path

RES = Path(__file__).parent / "results"
NAMES = {"ours": "ours (bf16, own kernels)", "llamacpp": "llama.cpp", "mlxlm": "mlx-lm", "vllm": "vllm-metal",
         "ours-noprefix": "ours, prefix cache off", "ours-q4": "ours, 4-bit", "ours-spec": "ours, spec k=4"}


def load(cfg, wl):
    p = RES / f"{cfg}_{wl}.json"
    return {l["concurrency"]: l for l in json.loads(p.read_text())["levels"]} if p.exists() else None


def table(configs, wl):
    data = {c: d for c in configs if (d := load(c, wl))}
    if not data:
        return ""
    concs = sorted({k for d in data.values() for k in d})
    out = [f"### {wl}\n", "Output tokens/s (higher is better):\n",
           "| engine | " + " | ".join(f"c={c}" for c in concs) + " |", "|---|" + "---:|" * len(concs)]
    for c, d in data.items():
        out.append(f"| {NAMES.get(c, c)} | " + " | ".join(
            f"{d[k]['output_tok_per_s']:.0f}" if k in d else "–" for k in concs) + " |")
    for metric, label in [("ttft_ms", "TTFT p50 / p99 (ms)"), ("itl_ms", "ITL p50 / p99 (ms)")]:
        out += ["", f"{label}:\n", "| engine | " + " | ".join(f"c={c}" for c in concs) + " |",
                "|---|" + "---:|" * len(concs)]
        for c, d in data.items():
            out.append(f"| {NAMES.get(c, c)} | " + " | ".join(
                f"{d[k][metric]['p50']:.0f} / {d[k][metric]['p99']:.0f}" if k in d else "–" for k in concs) + " |")
    bad = {c: sum(l["length_mismatches"] + l["failed"] for l in d.values()) for c, d in data.items()}
    if any(bad.values()):
        out.append(f"\nWarning: failed or wrong-length requests: {bad}")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    configs = sys.argv[1:] or list(NAMES)
    for wl in ("chat", "long", "prefix"):
        print(table(configs, wl))
