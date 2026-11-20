"""Speculative decoding on real models: greedy output == plain greedy output; rejection sampling sanity."""
import pytest
from engine.backend import MetalBackend
from engine.engine import Engine
from engine.model import Qwen3
from engine.scheduler import SamplingParams
from engine.spec import SpecEngine
from tests.test_engine import assert_match

TARGET, DRAFT = "Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def models():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TARGET)
    texts = ["Write a haiku about GPUs.", "def fibonacci(n):", "The three primary colors are",
             "Explain how a hash map works. " * 4, "Q: 12 * 12 = ? A:"]
    return Qwen3(TARGET, MetalBackend()), Qwen3(DRAFT, MetalBackend()), [tok.encode(t) for t in texts]


@pytest.mark.parametrize("k", [2, 5])
def test_greedy_spec_matches_plain(models, k):
    target, draft, prompts = models
    P = SamplingParams(max_tokens=48, ignore_eos=True)
    plain = Engine(target, 128).generate(prompts, P)
    eng = SpecEngine(target, draft, 128, k=k)
    spec = eng.generate(prompts, P)
    assert eng.blocks.num_free() == 128
    rate = eng.accepted / eng.drafted
    print(f"\nk={k}: acceptance {rate:.2f}, {eng.accepted + len(prompts) * 0} accepted of {eng.drafted} drafted")
    assert rate > 0.3
    assert_match(target, prompts, plain, spec)


def test_sampled_spec_with_identical_draft_accepts(models):
    _, draft, prompts = models
    eng = SpecEngine(draft, draft, 128, k=4)       # p == q (up to batch-shape numerics): accept ~always
    eng.generate(prompts, SamplingParams(max_tokens=40, temperature=0.8, ignore_eos=True))
    # not exactly 1: the draft scores with the decode kernel, the verify pass with the flash-prefill kernel
    assert eng.accepted / eng.drafted > 0.9, eng.accepted / eng.drafted


def test_rejection_sampling_preserves_target_distribution():
    """Emitted token distribution must be the target's p, not the draft's q (Leviathan et al., Thm 1)."""
    import random, types
    import mlx.core as mx
    import numpy as np
    from engine.backend import MLXBackend
    B = MLXBackend()
    eng = SpecEngine.__new__(SpecEngine)
    eng.B, eng.rng = B, random.Random(0)
    mx.random.seed(0)
    tl = mx.array([[2.0, 1.0, 0.5, 0.0, -1.0, 1.5], [0.0] * 6])        # target logits: row 0 + bonus row
    q = mx.softmax(mx.array([0.0, 2.0, 0.0, 1.0, 0.5, -1.0]))          # a deliberately different draft
    p = np.array(mx.softmax(tl[0]))
    r = types.SimpleNamespace(id=0, params=types.SimpleNamespace(temperature=1.0))
    qn = np.array(q)
    counts, n = np.zeros(6), 6000
    for d in np.random.default_rng(0).choice(6, size=n, p=qn / qn.sum()):
        keep, new = eng._accept(tl, [(r, 1)], {0: [int(d)]}, {0: [q]}, {0: 0})[0]
        counts[(keep + [new])[0]] += 1
    emp = counts / n
    assert np.abs(emp - p).max() < 0.025, (emp.round(3), p.round(3))
