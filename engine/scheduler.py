"""Paged KV block manager + continuous-batching scheduler (chunked prefill, preemption by recompute)."""
import itertools
from collections import OrderedDict, deque
from dataclasses import dataclass, field

_ids = itertools.count()


class BlockManager:
    """Free list + refcounts + (optional) prefix cache. A full, computed block is registered under
    key (parent block's key hash, its tokens); when its refcount drops to 0 it stays cached but evictable,
    and the free list is LRU so the least recently released block is reused (and evicted) first.
    Shared blocks are always full and never written again, so no copy-on-write is needed."""

    def __init__(self, num_blocks, block_size, prefix_caching=True):
        self.num_blocks, self.bs, self.prefix_caching = num_blocks, block_size, prefix_caching
        self.free = OrderedDict.fromkeys(range(num_blocks))    # LRU: popitem(last=False) = oldest
        self.ref = [0] * num_blocks
        self.key = [None] * num_blocks                         # block -> its cache key, if registered
        self.cached = {}                                       # cache key -> block
        self.hit_tokens = self.query_tokens = 0

    def num_free(self):
        return len(self.free)

    def needed(self, table, ctx_len):
        return max(0, -(-ctx_len // self.bs) - len(table))

    def grow(self, table, ctx_len):
        for _ in range(self.needed(table, ctx_len)):
            b, _ = self.free.popitem(last=False)
            assert self.ref[b] == 0
            if self.key[b] is not None:                        # evict its cached contents
                del self.cached[self.key[b]]
                self.key[b] = None
            self.ref[b] = 1
            table.append(b)

    def release(self, table):
        for b in reversed(table):                              # tail blocks become the oldest -> evicted first
            self.ref[b] -= 1
            assert self.ref[b] >= 0
            if self.ref[b] == 0:
                self.free[b] = None
        table.clear()

    def _keys(self, tokens, n_blocks):
        keys, parent = [], None
        for i in range(n_blocks):
            k = (hash(parent), tuple(tokens[i * self.bs:(i + 1) * self.bs]))
            keys.append(k)
            parent = k
        return keys

    def match(self, tokens):
        """Longest cached prefix of full blocks, always leaving >= 1 token to compute (so we get logits).
        Returns the blocks; the caller acquires them."""
        if not self.prefix_caching:
            return []
        hits = []
        for k in self._keys(tokens, (len(tokens) - 1) // self.bs):
            b = self.cached.get(k)
            if b is None:
                break
            hits.append(b)
        return hits

    def acquire(self, blocks, table):
        for b in blocks:
            if self.ref[b] == 0:
                del self.free[b]
            self.ref[b] += 1
            table.append(b)

    def register(self, table, tokens, num_computed):
        """Make the request's full, computed blocks findable by later requests."""
        if not self.prefix_caching:
            return
        n = num_computed // self.bs
        for i, k in enumerate(self._keys(tokens, n)):
            b = table[i]
            if self.key[b] is None and k not in self.cached:
                self.key[b], self.cached[k] = k, b


@dataclass
class SamplingParams:
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    ignore_eos: bool = False
    logit_bias: dict = None      # {token_id: bias}, OpenAI semantics


@dataclass(eq=False)   # identity semantics: hashable, and `r in running` doesn't compare token lists
class Request:
    prompt: list
    params: SamplingParams = field(default_factory=SamplingParams)
    id: int = field(default_factory=lambda: next(_ids))
    tokens: list = None          # prompt + generated
    num_computed: int = 0        # tokens whose KV is in the cache
    table: list = field(default_factory=list)
    done: bool = False
    finish_reason: str = None
    num_preemptions: int = 0

    def __post_init__(self):
        self.tokens = list(self.prompt)

    @property
    def output(self):
        return self.tokens[len(self.prompt):]

    def remaining(self):
        return len(self.tokens) - self.num_computed


class Scheduler:
    def __init__(self, blocks: BlockManager, max_batched_tokens=512, max_seqs=64):
        self.blocks, self.budget, self.max_seqs = blocks, max_batched_tokens, max_seqs
        self.waiting, self.running = deque(), []

    def add(self, r: Request):
        if not r.prompt:
            raise ValueError("empty prompt")
        if len(r.prompt) + r.params.max_tokens > self.blocks.num_blocks * self.blocks.bs:
            raise ValueError("request can never fit in the KV cache")
        self.waiting.append(r)

    def has_work(self):
        return bool(self.waiting or self.running)

    def _preempt(self, r, sched):
        """Free r's blocks and requeue it at the front; returns budget it had been given this step."""
        self.running.remove(r)
        self.blocks.release(r.table)
        r.num_computed = 0
        r.num_preemptions += 1
        self.waiting.appendleft(r)
        return sched.pop(r.id, (r, 0))[1]

    def schedule(self):
        """Returns {request_id: (request, num_new_tokens)}. Decodes get budget before prefill chunks;
        on OOM the most recently admitted request is preempted (its KV is recomputed later)."""
        sched, budget, preempted = {}, self.budget, False
        for r in sorted(self.running, key=lambda r: r.remaining() > 1):
            if r not in self.running:          # preempted earlier in this loop
                continue
            n = min(r.remaining(), budget)
            if n == 0:
                continue
            while self.blocks.needed(r.table, r.num_computed + n) > self.blocks.num_free():
                victim = self.running[-1]
                budget += self._preempt(victim, sched)
                preempted = True
                if victim is r:
                    break
            if r not in self.running:
                continue
            self.blocks.grow(r.table, r.num_computed + n)
            sched[r.id] = (r, n)
            budget -= n
        bm = self.blocks
        while self.waiting and budget > 0 and len(self.running) < self.max_seqs and not preempted:
            r = self.waiting[0]
            hits = bm.match(r.tokens)
            computed = len(hits) * bm.bs
            n = min(len(r.tokens) - computed, budget)
            idle_hits = sum(1 for b in hits if bm.ref[b] == 0)   # acquiring these shrinks the free list
            if -(-(computed + n) // bm.bs) - len(hits) + idle_hits > bm.num_free():
                break
            self.waiting.popleft()
            self.running.append(r)
            bm.acquire(hits, r.table)
            r.num_computed = computed
            bm.hit_tokens += computed
            bm.query_tokens += len(r.tokens)
            bm.grow(r.table, r.num_computed + n)
            sched[r.id] = (r, n)
            budget -= n
        return sched

    def finish(self, r, reason):
        r.done, r.finish_reason = True, reason
        if r in self.running:
            self.running.remove(r)
        elif r in self.waiting:
            self.waiting.remove(r)
        self.blocks.release(r.table)
