"""Scheduler/block-manager stress test with a fake model under seeded random workloads.
Fake KV at position p = hash(KV at p-1, token p), read back through the block table, so like real K/V it
depends on the whole prefix; next token = KV of the last position. Any slot-mapping / preemption /
prefix-cache bookkeeping bug (including sharing a block whose earlier context differs) changes the output."""
import random, struct, zlib
from collections import Counter
import numpy as np
import pytest
from engine.engine import Engine
from engine.scheduler import SamplingParams

V, EOS = 1000, 7


def chain(prev, tok):
    return zlib.crc32(struct.pack("<qq", prev, tok))


class FakeB:
    def alloc_kv(self, n_layers, num_slots, n_kv, d):
        return np.full(num_slots, -1, np.int64)

    def sample(self, logits, *a):
        return logits

    def tolist(self, x):
        return list(x)

    def eval(self, *x):
        pass

    def take_rows(self, x, rows):
        return np.asarray(x)[list(rows)]


class FakeModel:
    B, n_layers, Hkv, D = FakeB(), 1, 1, 1
    cfg = {"vocab_size": V}

    def __init__(self, noise=0.0):
        self.noise = noise          # draft only: probability of proposing a wrong token

    def forward(self, batch, cache, bs):
        assert sorted(s.q_start for s in batch.seqs) == [s.q_start for s in batch.seqs]
        out = []
        for s in batch.seqs:
            slot = lambda p: s.block_table[p // bs] * bs + p % bs
            for i in range(s.q_len):
                p = s.ctx_len - s.q_len + i
                assert batch.slot_mapping[s.q_start + i] == slot(p) and batch.positions[s.q_start + i] == p
                prev = cache[slot(p - 1)] if p > 0 else 0
                assert prev >= 0, "reading an unwritten slot"
                cache[slot(p)] = chain(prev, batch.tokens[s.q_start + i])
            for i in range(s.q_len):
                if s.q_start + i in batch.logits_rows:
                    h = int(cache[slot(s.ctx_len - s.q_len + i)])
                    wrong = self.noise and (h >> 11) % 1000 < self.noise * 1000
                    out.append((h + wrong) % V)
        return np.asarray(out, np.int64)


def reference(prompt, max_tokens):
    h, out = 0, []
    for t in prompt:
        h = chain(h, t)
    while len(out) < max_tokens:
        t = h % V
        out.append(t)
        h = chain(h, t)
        if t == EOS:
            break
    return out


def check_invariants(eng, sched_out):
    bm, sc = eng.blocks, eng.sched
    owners = Counter(b for r in sc.running for b in r.table)
    assert all(bm.ref[b] == owners.get(b, 0) for b in range(bm.num_blocks)), "refcount mismatch"
    assert len(set(bm.free)) == len(bm.free) == sum(1 for c in bm.ref if c == 0), "free list corrupt"
    for r in sc.running:
        assert len(r.table) == -(-r.num_computed // bm.bs), "table size != computed tokens"
    for r in sc.waiting:
        assert not r.table and r.num_computed == 0
    assert all(bm.key[b] == k for k, b in bm.cached.items()), "cache map -> block mismatch"
    assert sum(k is not None for k in bm.key) == len(bm.cached), "stale block keys"
    assert sum(n for _, n in sched_out.values()) <= sc.budget
    assert len(sc.running) <= sc.max_seqs


@pytest.mark.parametrize("seed", range(150))
def test_random_workload(seed):
    run_workload(seed)


@pytest.mark.parametrize("seed", range(60))
@pytest.mark.parametrize("noise", [0.0, 0.4])
def test_speculative_workload(seed, noise):
    eng = run_workload(seed, draft=FakeModel(noise))
    if eng.drafted:
        rate = eng.accepted / eng.drafted
        assert rate == 1.0 if noise == 0 else 0 < rate < 1, rate
        SPEC.append(eng.drafted)


def test_workloads_exercise_speculation():
    assert sum(1 for d in SPEC if d) > 40, SPEC


SPEC = []


def run_workload(seed, draft=None):
    rng = random.Random(seed)
    bs = rng.choice([1, 2, 4, 16])
    num_blocks = rng.randint(4, 24) * (16 // bs)
    cap = num_blocks * bs
    kw = dict(block_size=bs, max_batched_tokens=rng.choice([8, 32, 512]), max_seqs=rng.choice([1, 4, 64]),
              eos_ids=[EOS], prefix_caching=seed % 4 != 0)
    if draft is None:
        eng = Engine(FakeModel(), num_blocks, **kw)
    else:
        from engine.spec import SpecEngine
        eng = SpecEngine(FakeModel(), draft, num_blocks, k=rng.choice([1, 2, 4, 7]), **kw)
    orig, log = eng.sched.schedule, {}

    def spy():
        log["last"] = orig()
        return log["last"]
    eng.sched.schedule = spy

    pending, prefixes, seen = [], [[rng.randrange(V) for _ in range(rng.randint(1, 80))] for _ in range(3)], []
    for _ in range(rng.randint(1, 30)):
        if seen and rng.random() < 0.15:                      # exact duplicate prompt: full-hit path
            prompt = list(rng.choice(seen))
        else:
            L = max(1, rng.choice([1, bs - 1, bs, bs + 1, rng.randint(1, 120)]))
            pre = rng.choice(prefixes) if rng.random() < 0.6 else []
            prompt = (pre + [rng.randrange(V) for _ in range(L)])[:cap - 1]
        seen.append(prompt)
        pending.append((rng.randint(0, 10), prompt, rng.randint(1, min(64, cap - len(prompt)))))
    reqs, step = [], 0
    while pending or eng.sched.has_work():
        for item in [p for p in pending if p[0] <= step]:
            pending.remove(item)
            reqs.append((eng.add(item[1], SamplingParams(max_tokens=item[2])), item))
        eng.step()
        check_invariants(eng, log.get("last", {}))
        step += 1
        assert step < 100_000, "livelock"
    assert eng.blocks.num_free() == num_blocks, "blocks leaked"
    for r, (_, prompt, mt) in reqs:
        assert r.done and not r.table
        assert r.output == reference(prompt, mt)
    if draft is None:
        PREEMPTIONS.append(sum(r.num_preemptions for r, _ in reqs))
        HITS.append(eng.blocks.hit_tokens)
    return eng


PREEMPTIONS, HITS = [], []


def test_workloads_exercise_prefix_cache():
    print(f"\n{sum(1 for h in HITS if h)}/{len(HITS)} seeds hit the prefix cache, {sum(HITS)} tokens reused")
    assert sum(1 for h in HITS if h) > 50


def test_workloads_exercise_preemption():
    print(f"\n{sum(1 for p in PREEMPTIONS if p)}/{len(PREEMPTIONS)} seeds preempted, {sum(PREEMPTIONS)} preemptions")
    assert sum(1 for p in PREEMPTIONS if p) > 50
