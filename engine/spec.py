"""Speculative decoding (Leviathan et al.): a small draft model proposes k tokens per decoding request,
the target scores [last token + k drafts] in one pass, and the longest acceptable prefix is kept.

The draft shares the target's block tables (it only needs the same slot count; its own KV geometry may
differ), so its cache is a second KV pool indexed by the same slots. Rejected positions are rolled back by
lowering num_computed and releasing surplus blocks; stale KV past num_computed is never read before it is
overwritten. Greedy requests accept while draft == target argmax; sampled requests use rejection sampling,
which keeps the output distribution exactly the target's."""
import random
import numpy as np
from .engine import Engine


class SpecEngine(Engine):
    def __init__(self, model, draft, num_blocks, k=4, seed=0, **kw):
        super().__init__(model, num_blocks, **kw)
        assert getattr(draft, "cfg", {}).get("vocab_size") == getattr(model, "cfg", {}).get("vocab_size")
        self.draft, self.k, self.rng = draft, k, random.Random(seed)
        self.dcache = draft.B.alloc_kv(draft.n_layers, num_blocks * self.bs, draft.Hkv, draft.D)
        self.drafted = self.accepted = 0

    def _spec_k(self, r, n):
        """Tokens to draft for r this step (0 = plain decode). Reserves blocks for the verify pass.
        ponytail: temperature-only sampling; top-p/top-k/logit_bias requests decode without drafts."""
        p = r.params
        # decoding = exactly the last token is uncomputed (a re-prefill after preemption can also get n == 1)
        if n != 1 or r.remaining() != 1 or not r.output or p.top_p < 1 or p.top_k or p.logit_bias:
            return 0
        k = min(self.k, p.max_tokens - len(r.output) - 1)
        if k <= 0 or self.blocks.needed(r.table, len(r.tokens) + k) > self.blocks.num_free():
            return 0
        self.blocks.grow(r.table, len(r.tokens) + k)
        return k

    def _draft(self, spec):
        """k rounds of the draft model. Round 0 feeds every token the draft hasn't seen (catch-up chunk),
        later rounds feed the previous draft token. Returns {id: drafts}, {id: [draft prob rows or None]}."""
        B, bs = self.draft.B, self.bs
        drafts = {r.id: [] for r, _ in spec}
        probs = {r.id: [] for r, _ in spec}
        for r, _ in spec:
            if getattr(r, "draft_epoch", None) != r.num_preemptions:   # its blocks were freed: recompute
                r.draft_epoch, r.draft_computed = r.num_preemptions, 0
        for i in range(max(k for _, k in spec)):
            views = [_View(r, r.tokens + drafts[r.id], r.draft_computed) for r, k in spec if k > i]
            work = sorted(((v, len(v.tokens) - v.num_computed) for v in views), key=lambda vn: vn[1] != 1)
            batch, samp = self.build(work, bs)
            logits = self.draft.forward(batch, self.dcache, bs)
            temps = [v.r.params.temperature for v in samp]
            toks = B.tolist(B.sample(logits, temps, [1.0] * len(samp), [0] * len(samp)))
            q = B.softmax_t(logits, temps) if any(temps) else None
            for j, (v, t) in enumerate(zip(samp, toks)):
                drafts[v.r.id].append(t)
                probs[v.r.id].append(q[j] if temps[j] else None)
                v.r.draft_computed = len(v.tokens)
        return drafts, probs

    def step(self):
        sched = self.sched.schedule()
        if not sched:
            return []
        work = list(sched.values())
        spec = [(r, k) for r, n in work if (k := self._spec_k(r, n))]
        drafts, probs = self._draft(spec) if spec else ({}, {})
        L0 = {r.id: len(r.tokens) for r, _ in spec}
        for r, _ in spec:                                    # provisional: verify [last token + drafts]
            r.tokens += drafts[r.id]
        work = sorted(((r, n + len(drafts.get(r.id, ()))) for r, n in work), key=lambda rn: rn[1] != 1)
        is_spec = {r for r, _ in spec}
        batch, samp = self.build(work, self.bs, all_rows=is_spec)
        logits = self.model.forward(batch, self.cache, self.bs)
        first_row, i = {}, 0                                 # build() emits logits rows in work order
        for r, n in work:
            if r in is_spec:
                first_row[r.id] = i; i += n
            elif r in samp:
                first_row[r.id] = i; i += 1
        B = self.B
        nxt = self.sample(B.take_rows(logits, [first_row[r.id] for r in samp]), samp) if samp else []
        acc = self._accept(logits, spec, drafts, probs, first_row) if spec else {}
        if not samp and not spec:
            B.eval(logits)
        out = []
        for r, n in work:
            if r not in is_spec:
                r.num_computed += n
                self.blocks.register(r.table, r.tokens, r.num_computed)
                continue
            L = L0[r.id]
            del r.tokens[L:]                                 # drop the provisional drafts
            keep, new = acc[r.id]
            r.num_computed = L + len(keep)                   # KV valid through the last accepted draft
            r.draft_computed = min(r.draft_computed, r.num_computed)
            surplus = r.table[-(-r.num_computed // self.bs):]
            del r.table[-(-r.num_computed // self.bs):]
            self.blocks.release(surplus)
            self.blocks.register(r.table, r.tokens + keep, r.num_computed)
            self.drafted += len(drafts[r.id]); self.accepted += len(keep)
            out += self.commit(r, keep + [new])
        for r, t in zip(samp, nxt):
            out += self.commit(r, [t])
        return out

    def _accept(self, logits, spec, drafts, probs, first_row):
        B, res = self.B, {}
        greedy = [(r, k) for r, k in spec if r.params.temperature == 0]
        if greedy:
            rows = [first_row[r.id] + j for r, k in greedy for j in range(k + 1)]
            n = len(rows)
            tgt = B.tolist(B.sample(B.take_rows(logits, rows), [0] * n, [1.0] * n, [0] * n))
            o = 0
            for r, k in greedy:
                t, d, j = tgt[o:o + k + 1], drafts[r.id], 0
                while j < k and d[j] == t[j]:
                    j += 1
                res[r.id] = (d[:j], t[j])
                o += k + 1
        for r, k in spec:
            if r.params.temperature == 0:
                continue
            p = B.softmax_t(B.take_rows(logits, range(first_row[r.id], first_row[r.id] + k + 1)),
                            [r.params.temperature] * (k + 1))
            d, q = drafts[r.id], probs[r.id]
            ratio = B.tolist(B.accept_ratios(p, q, d))      # p_i(d_i) / q_i(d_i)
            j = 0
            while j < k and self.rng.random() < ratio[j]:
                j += 1
            dist = p[j] if j == k else B.residual(p[j], q[j])   # norm(max(p - q, 0)) after a rejection
            res[r.id] = (d[:j], B.tolist(B.sample_probs(dist))[0])
        return res


class _View:
    """Stand-in request for the draft pass: same block table, draft-side token/computed counters."""
    def __init__(self, r, tokens, num_computed):
        self.r, self.tokens, self.num_computed, self.table = r, tokens, num_computed, r.table
