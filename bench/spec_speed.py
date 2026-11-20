"""Speculative decoding: decode tok/s and acceptance vs plain decoding (greedy), Qwen3-0.6B drafting for 1.7B.

    uv run python bench/spec_speed.py"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer
from engine.backend import MetalBackend
from engine.engine import Engine
from engine.model import Qwen3
from engine.scheduler import SamplingParams
from engine.spec import SpecEngine

TARGET, DRAFT = "Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B"
TEXTS = ["Explain how TCP congestion control works.", "Write a Python function that merges two sorted lists.",
         "Summarize the causes of the French Revolution.", "What is the difference between a process and a thread?",
         "Describe how a transformer language model generates text.", "Give me a recipe for pancakes.",
         "List the planets of the solar system with one fact each.", "How does public-key cryptography work?"]


def run(eng, prompts, n_out):
    P = SamplingParams(max_tokens=n_out, ignore_eos=True)
    reqs = [eng.add(p, P) for p in prompts]
    while any(r.num_computed < len(r.prompt) for r in reqs):   # exclude prefill from the timing
        eng.step()
    produced0 = sum(len(r.output) for r in reqs)
    t0 = time.perf_counter()
    while eng.sched.has_work():
        eng.step()
    return (sum(len(r.output) for r in reqs) - produced0) / (time.perf_counter() - t0)


def main(quant=None):
    tok = AutoTokenizer.from_pretrained(TARGET)
    chat = lambda t: tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True,
                                             tokenize=False, enable_thinking=False)
    prompts = [tok.encode(chat(t)) for t in TEXTS]
    target, draft = Qwen3(TARGET, MetalBackend(), quant=quant), Qwen3(DRAFT, MetalBackend(), quant=quant)
    print(f"target {TARGET} / draft {DRAFT}, weights {'4-bit' if quant else 'bf16'}, greedy, 128 new tokens")
    for bsz in (1, 4, 16):
        ps = (prompts * 2)[:bsz]
        base = run(Engine(target, 512), ps, 128)
        line = f"  batch {bsz:2d}: plain {base:6.1f} tok/s"
        for k in (2, 4, 6):
            eng = SpecEngine(target, draft, 512, k=k)
            t = run(eng, ps, 128)
            line += f" | k={k} {t:6.1f} ({t / base:.2f}x, acc {eng.accepted / max(1, eng.drafted):.2f})"
        print(line)


if __name__ == "__main__":
    main(quant=4 if "--q4" in sys.argv else None)
