"""Engine: owns model + KV cache + scheduler. step() runs one continuous-batching iteration."""
from .model import Batch, SeqSlice
from .scheduler import BlockManager, Scheduler, Request, SamplingParams


class Engine:
    def __init__(self, model, num_blocks, block_size=16, max_batched_tokens=512, max_seqs=64, eos_ids=(),
                 prefix_caching=True):
        self.model, self.B, self.bs = model, model.B, block_size
        self.cache = self.B.alloc_kv(model.n_layers, num_blocks * block_size, model.Hkv, model.D)
        self.blocks = BlockManager(num_blocks, block_size, prefix_caching)
        self.sched = Scheduler(self.blocks, max_batched_tokens, max_seqs)
        self.eos = set(eos_ids)

    @staticmethod
    def from_pretrained(model_id, backend, kv_gb=None, reserve_gb=1.5, quant=None, draft=None, spec_k=4, **kw):
        """Load weights, then size the KV cache from the GPU memory actually left (macOS only lets the GPU
        wire part of RAM). With a draft model, its KV pool is the same size and is budgeted too."""
        import json
        from .model import Qwen3
        from .weights import resolve
        m = Qwen3(model_id, backend, quant=quant)
        d = Qwen3(draft, type(backend)(), quant=quant) if draft else None
        per_token = m.kv_bytes_per_token + (d.kv_bytes_per_token if d else 0)
        if kv_gb is None:
            kv_gb = max(0.25, backend.free_gpu_bytes() / 2**30 - reserve_gb) * m.kv_bytes_per_token / per_token
        bs = kw.get("block_size", 16)
        num_blocks = int(kv_gb * 2**30 // (m.kv_bytes_per_token * bs))
        gen = json.loads((resolve(model_id) / "generation_config.json").read_text())
        eos = gen.get("eos_token_id", [])
        kw["eos_ids"] = eos if isinstance(eos, list) else [eos]
        if d:
            from .spec import SpecEngine
            return SpecEngine(m, d, num_blocks, k=spec_k, **kw)
        return Engine(m, num_blocks, **kw)

    def add(self, prompt, params=None):
        r = Request(list(prompt), params or SamplingParams())
        self.sched.add(r)
        return r

    def step(self):
        """One iteration. Returns [(request, new_token)] for requests that produced a token."""
        sched = self.sched.schedule()
        if not sched:
            return []
        work = sorted(sched.values(), key=lambda rn: rn[1] != 1)  # q_len==1 rows first (backend layout)
        batch, samp = self.build(work, self.bs)
        logits = self.model.forward(batch, self.cache, self.bs)
        nxt = self.sample(logits, samp)
        for r, n in work:
            r.num_computed += n
            self.blocks.register(r.table, r.tokens, r.num_computed)   # before commit can free the table
        out = []
        for r, t in zip(samp, nxt):
            out += self.commit(r, [t])
        return out

    @staticmethod
    def build(work, bs, all_rows=()):
        """Flatten [(request, n_new)] into a Batch. Logits for each chunk's last row when it reaches the end
        of the request's tokens (those requests are returned as `samp`), or for every row of requests in
        all_rows (speculative verification)."""
        toks, pos, slots, seqs, rows, samp = [], [], [], [], [], []
        for r, n in work:
            s0 = r.num_computed
            seqs.append(SeqSlice(len(toks), n, s0 + n, r.table))
            if r in all_rows:
                rows += range(len(toks), len(toks) + n)
            for p in range(s0, s0 + n):
                toks.append(r.tokens[p]); pos.append(p)
                slots.append(r.table[p // bs] * bs + p % bs)
            if r not in all_rows and s0 + n == len(r.tokens):
                rows.append(len(toks) - 1); samp.append(r)
        return Batch(toks, pos, slots, seqs, rows), samp

    def sample(self, logits, samp):
        """Sample one token per request in samp from its logits row (rows in the same order)."""
        if not samp:  # nothing to sample, but the KV writes are side effects of this graph: it must still run
            self.B.eval(logits)
            return []
        P = [r.params for r in samp]
        bias = [(i, t, b) for i, p in enumerate(P) for t, b in (p.logit_bias or {}).items()]
        if bias:
            logits = self.B.add_bias(logits, *zip(*bias))
        return self.B.tolist(self.B.sample(logits, [p.temperature for p in P], [p.top_p for p in P],
                                           [p.top_k for p in P]))

    def commit(self, r, toks):
        """Append generated tokens, stopping at EOS / max_tokens. Returns [(r, t)] actually emitted."""
        out = []
        for t in toks:
            r.tokens.append(t)
            out.append((r, t))
            if not r.params.ignore_eos and t in self.eos:
                self.sched.finish(r, "stop")
                break
            if len(r.output) >= r.params.max_tokens:
                self.sched.finish(r, "length")
                break
        return out

    def generate(self, prompts, params=None):
        """Offline helper: run everything to completion, return output token lists."""
        reqs = [self.add(p, params) for p in prompts]
        while self.sched.has_work():
            self.step()
        return [r.output for r in reqs]
