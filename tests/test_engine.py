"""Phase 3 acceptance on the real model: batched output == single-request output, no leaked blocks."""
import os
import pytest
from engine.backend import MLXBackend, MetalBackend
from engine.engine import Engine
from engine.model import Qwen3
from engine.scheduler import SamplingParams

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")
Backend = {"mlx": MLXBackend, "metal": MetalBackend}[os.environ.get("BACKEND", "metal")]
P = SamplingParams(max_tokens=32, ignore_eos=True)


@pytest.fixture(scope="module")
def setup():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    texts = ["Hi", "Write a haiku about GPUs.", "List three prime numbers and explain why they are prime. " * 3,
             "The history of the Roman empire, briefly: " * 12, "def fibonacci(n):", "Q: 2+2=? A:"]
    prompts = [tok.encode(t) for t in texts]
    return Qwen3(MODEL, Backend()), prompts


def run(m, prompts, num_blocks, max_batched_tokens, max_seqs=64):
    eng = Engine(m, num_blocks, 16, max_batched_tokens, max_seqs)
    outs = eng.generate(prompts, P)
    assert eng.blocks.num_free() == num_blocks
    return outs, eng


def assert_match(m, prompts, ref, got, tie=0.25):
    """Exact match, except a divergence is allowed at a bf16 near-tie (|logit gap| <= tie, i.e. ~2 ulp
    at |logit|~20); comparison stops there (same criterion as vLLM's check_logprobs_close)."""
    from tests.test_model import run_seq
    for p, a, b in zip(prompts, ref, got):
        d = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        if d is None:
            assert a == b
            continue
        cache = m.B.alloc_kv(m.n_layers, 64 * 16, m.Hkv, m.D)
        l = run_seq(m, cache, p + a[:d], 0, list(range(64)))
        gap = abs(l[a[d]].item() - l[b[d]].item())
        assert gap <= tie and l.max().item() - l[b[d]].item() <= tie, (d, gap)


def test_batched_equals_single(setup):
    m, prompts = setup
    single = [run(m, [p], 64, 512)[0][0] for p in prompts]
    batched, _ = run(m, prompts, 256, 512)
    chunked, _ = run(m, prompts, 256, 24)          # forces chunked prefill mixed with decodes
    assert_match(m, prompts, single, batched)
    assert_match(m, prompts, single, chunked)


def test_preemption_recompute(setup):
    m, prompts = setup
    single = [run(m, [p], 64, 512)[0][0] for p in prompts]
    eng = Engine(m, 14, 16, 64, 64)                  # 14 blocks * 16 = 224 slots: must preempt
    reqs = [eng.add(p, P) for p in prompts]
    while eng.sched.has_work():
        eng.step()
    assert eng.blocks.num_free() == 14
    assert sum(r.num_preemptions for r in reqs) > 0
    outs = [r.output for r in reqs]
    assert_match(m, prompts, single, outs)


def test_prefix_cache_matches_uncached(setup):
    m, prompts = setup
    system = prompts[3]                                  # ~100-token shared prefix
    shared = [system + p for p in prompts] + [system + prompts[1]]   # incl. an exact repeat
    cold, _ = run(m, shared, 256, 512)
    eng = Engine(m, 256, 16, 512, 64, prefix_caching=False)
    ref = eng.generate(shared, P)
    eng = Engine(m, 256, 16, 512, 64, prefix_caching=True)
    eng.generate(shared[:1], P)                          # warm the cache
    hot = eng.generate(shared, P)
    assert eng.blocks.hit_tokens >= len(system) // 16 * 16 * len(shared)
    assert eng.blocks.num_free() == 256
    assert_match(m, shared, ref, hot)


def test_logit_bias(setup):
    m, prompts = setup
    eng = Engine(m, 64, 16, 512, 64)
    out = eng.generate([prompts[0]], SamplingParams(max_tokens=5, logit_bias={42: 100.0}))
    assert out == [[42] * 5]
