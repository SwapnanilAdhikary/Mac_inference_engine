"""Phase 2 acceptance: our logits and greedy output match mlx-lm (reference only, never imported by engine/)."""
import os
import mlx.core as mx
import numpy as np
import pytest
from engine.backend import MLXBackend, MetalBackend
from engine.model import Qwen3, Batch, SeqSlice

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")
Backend = {"mlx": MLXBackend, "metal": MetalBackend}[os.environ.get("BACKEND", "metal")]
BS = 16


@pytest.fixture(scope="module")
def models():
    from mlx_lm import load
    ref, tok = load(MODEL)
    return Qwen3(MODEL, Backend()), ref, tok


def run_seq(m, cache, ids, start, table):
    """Feed ids as one chunk at positions start.. for one sequence; returns last-token logits."""
    n = len(ids)
    pos = list(range(start, start + n))
    slots = [table[p // BS] * BS + p % BS for p in pos]
    b = Batch(ids, pos, slots, [SeqSlice(0, n, start + n, table)], [n - 1])
    out = m.forward(b, cache, BS)
    m.B.eval(out)  # in-place KV writes only happen when the graph runs
    return out[0]


def ref_logits(ref, ids):
    return ref(mx.array([ids]))[0, -1].astype(mx.float32)


@pytest.mark.parametrize("n", [1, BS - 1, BS, BS + 1, 100])
def test_prefill_logits_match(models, n):
    m, ref, tok = models
    ids = tok.encode("The quick brown fox jumps over the lazy dog. " * 20)[:n]
    cache = m.B.alloc_kv(m.n_layers, 64 * BS, m.Hkv, m.D)
    ours, theirs = run_seq(m, cache, ids, 0, list(range(64))[::-1]), ref_logits(ref, ids)
    cos = (ours * theirs).sum() / (mx.linalg.norm(ours) * mx.linalg.norm(theirs))
    assert cos.item() > 0.999, cos.item()
    assert mx.argmax(ours).item() == mx.argmax(theirs).item()


def test_greedy_matches(models):
    from mlx_lm.models.cache import make_prompt_cache
    m, ref, tok = models
    prompt = tok.apply_chat_template([{"role": "user", "content": "Explain paged attention in two sentences."}],
                                     add_generation_prompt=True, enable_thinking=False)
    steps = 48
    # reference greedy loop (keep its logits to judge any divergence)
    rc, out_ref, ref_logits_seq = make_prompt_cache(ref), [], []
    logits = ref(mx.array([prompt]), cache=rc)[0, -1]
    for _ in range(steps):
        ref_logits_seq.append(logits.astype(mx.float32))
        t = mx.argmax(logits).item(); out_ref.append(t)
        logits = ref(mx.array([[t]]), cache=rc)[0, -1]
    # ours: chunked prefill (chunks of 7 to cross block boundaries), then decode
    cache, table = m.B.alloc_kv(m.n_layers, 64 * BS, m.Hkv, m.D), list(np.random.default_rng(0).permutation(64))
    for s in range(0, len(prompt), 7):
        logits = run_seq(m, cache, prompt[s:s + 7], s, table)
    out, pos = [], len(prompt)
    for _ in range(steps):
        t = mx.argmax(logits).item(); out.append(t)
        logits = run_seq(m, cache, [t], pos, table); pos += 1
    print("\nours:", tok.decode(out), "\nref: ", tok.decode(out_ref))
    d = next((i for i, (a, b) in enumerate(zip(out, out_ref)) if a != b), None)
    if d is not None:  # allowed only at a bf16 near-tie in the *reference's* logits, and not early
        l = ref_logits_seq[d]
        gap = (l[out_ref[d]] - l[out[d]]).item()
        print(f"diverged at {d}: reference logit gap {gap:.4f}")
        assert d >= 16 and gap <= 0.25, (d, gap)
