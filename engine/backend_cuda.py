"""CUDA backend: same interface as backend.MLXBackend, built on torch (eager) with our own paged-attention
kernels (Triton, and a CUDA C++ port). The engine, scheduler, model and server are unchanged.

    Backend: TorchBackend(attn="triton" | "cuda" | "torch")   ("torch" = copy-KV-then-SDPA baseline)
Status: written on a Mac without an NVIDIA GPU; tests/test_cuda.py is the first thing to run on the box."""
import numpy as np
import torch
import torch.nn.functional as F

_DT = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32, "i32": torch.int32}


class TorchBackend:
    name = "cuda"

    def __init__(self, device="cuda", attn="triton"):
        self.dev, self.attn = torch.device(device), attn
        self._inv = {}

    # ---- arrays -------------------------------------------------------------
    def asarray(self, a, dt=None):
        a = np.ascontiguousarray(a)
        if dt == "bf16":  # raw uint16 bits from the safetensors reader
            return torch.from_numpy(a.view(np.int16)).view(torch.bfloat16).to(self.dev)
        t = torch.from_numpy(a)
        if t.dtype == torch.int32:  # indices: torch indexing wants int64
            t = t.long()
        return t.to(self.dev, _DT[dt]) if dt else t.to(self.dev)

    def concat(self, xs, axis=0):
        return torch.cat(xs, dim=axis)

    def zeros(self, shape, dt="bf16"):
        return torch.zeros(shape, dtype=_DT[dt], device=self.dev)

    def eval(self, *xs):
        torch.cuda.synchronize(self.dev)

    def tolist(self, x):
        return x.tolist()

    def free_gpu_bytes(self):
        return torch.cuda.mem_get_info(self.dev)[0]

    # ---- layers -------------------------------------------------------------
    def embedding(self, table, ids):
        return table[ids]

    def linear(self, x, w):
        return F.linear(x, w)

    def logits(self, h, w):
        return F.linear(h, w).float()

    def rms_norm(self, x, w, eps):
        xf = x.float()
        return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)

    def add_rms_norm(self, x, delta, w, eps):
        x = x + delta
        return self.rms_norm(x, w, eps), x

    def rope(self, x, positions, theta):
        d = x.shape[-1]
        if (theta, d) not in self._inv:
            self._inv[theta, d] = theta ** (-torch.arange(0, d, 2, dtype=torch.float32, device=self.dev) / d)
        ang = positions.float()[:, None] * self._inv[theta, d][None]
        cos, sin = ang.cos()[:, None], ang.sin()[:, None]
        xf = x.float()
        x1, x2 = xf[..., : d // 2], xf[..., d // 2:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).to(x.dtype)

    def silu_mul(self, gu):
        g, u = gu.chunk(2, dim=-1)
        return F.silu(g.float()).to(g.dtype) * u

    # ---- KV cache + attention ----------------------------------------------
    def alloc_kv(self, num_layers, num_slots, n_kv, head_dim, dt="bf16"):
        return [[torch.zeros(num_slots, n_kv, head_dim, dtype=_DT[dt], device=self.dev) for _ in range(2)]
                for _ in range(num_layers)]

    def prepare_attention(self, batch, block_size):
        seqs = batch.seqs
        dec = [s for s in seqs if s.q_len == 1]
        pre = [s for s in seqs if s.q_len > 1]
        assert seqs[:len(dec)] == dec, "runner must place decode seqs before prefill seqs"
        L = lambda xs: torch.tensor(xs, dtype=torch.long, device=self.dev)
        meta = {"slots": L(batch.slot_mapping), "pos": L(batch.positions), "bs": block_size, "n_dec": len(dec)}
        if dec:
            maxb = max(len(s.block_table) for s in dec)
            bt = np.zeros((len(dec), maxb), np.int32)
            for i, s in enumerate(dec):
                bt[i, :len(s.block_table)] = s.block_table
            meta["dec"] = (torch.from_numpy(bt).to(self.dev),
                           torch.tensor([s.ctx_len for s in dec], dtype=torch.int32, device=self.dev),
                           max(s.ctx_len for s in dec))
        meta["pre"] = []
        for s in pre:  # ponytail: prefill gathers its KV and uses SDPA (flash) per sequence; paged prefill next
            j = np.arange(s.ctx_len)
            slots = np.asarray(s.block_table)[j // block_size] * block_size + j % block_size
            meta["pre"].append((s.q_start, s.q_len, s.ctx_len, L(slots)))
        return meta

    def attention_block(self, qkv, qn, kn, cache, meta, H, Hkv, D, eps, theta):
        T = qkv.shape[0]
        q = self.rope(self.rms_norm(qkv[:, :H * D].view(T, H, D), qn, eps), meta["pos"], theta)
        k = self.rope(self.rms_norm(qkv[:, H * D:(H + Hkv) * D].view(T, Hkv, D), kn, eps), meta["pos"], theta)
        kc, vc = cache
        kc[meta["slots"]] = k                                  # in place: torch is eager
        vc[meta["slots"]] = qkv[:, (H + Hkv) * D:].view(T, Hkv, D)
        out = []
        S = meta["n_dec"]
        if "dec" in meta:
            bt, cl, max_ctx = meta["dec"]
            out.append(self.decode_attention(q[:S], kc, vc, bt, cl, max_ctx, meta["bs"]))
        for q0, ql, ctx, slots in meta["pre"]:
            kk, vv = kc[slots].transpose(0, 1), vc[slots].transpose(0, 1)        # [Hkv, ctx, D]
            qq = q[q0:q0 + ql].transpose(0, 1)                                    # [H, ql, D]
            mask = torch.ones(ql, ctx, dtype=torch.bool, device=self.dev).tril(ctx - ql)
            o = F.scaled_dot_product_attention(qq[None], kk[None], vv[None], attn_mask=mask, enable_gqa=True)
            out.append(o[0].transpose(0, 1))
        return torch.cat(out, 0).reshape(T, H * D)

    def decode_attention(self, q, kc, vc, block_tables, ctx_lens, max_ctx, bs):
        """q [S, H, D] -> [S, H, D] over the paged cache."""
        if self.attn == "triton":
            from .kernels_triton import paged_decode
            return paged_decode(q, kc, vc, block_tables, ctx_lens, bs)
        if self.attn == "cuda":
            from .kernels_cuda import paged_decode
            return paged_decode(q, kc, vc, block_tables, ctx_lens, bs)
        return paged_decode_reference(q, kc, vc, block_tables, ctx_lens, bs)

    # ---- sampling -----------------------------------------------------------
    def add_bias(self, logits, rows, toks, vals):
        logits = logits.clone()
        logits[torch.tensor(rows, device=self.dev), torch.tensor(toks, device=self.dev)] += torch.tensor(
            vals, device=self.dev)
        return logits

    def take_rows(self, x, rows):
        return x[torch.tensor(list(rows), dtype=torch.long, device=self.dev)]

    def sample(self, logits, temps, top_ps, top_ks, key=None):
        logits = logits.float()
        greedy = logits.argmax(-1)
        if all(t == 0 for t in temps):
            return greedy
        t = torch.tensor([max(t, 1e-5) for t in temps], device=self.dev)[:, None]
        sl, order = (logits / t).sort(-1, descending=True)
        p = sl.softmax(-1)
        cum = p.cumsum(-1) - p
        rank = torch.arange(sl.shape[-1], device=self.dev)[None]
        k = torch.tensor([k if k > 0 else sl.shape[-1] for k in top_ks], device=self.dev)[:, None]
        keep = (cum < torch.tensor(top_ps, device=self.dev)[:, None]) & (rank < k)
        pick = torch.multinomial(torch.where(keep, sl, -torch.inf).softmax(-1), 1)
        sampled = order.gather(-1, pick)[:, 0]
        return torch.where(torch.tensor([t == 0 for t in temps], device=self.dev), greedy, sampled)


def paged_decode_reference(q, kc, vc, block_tables, ctx_lens, bs):
    """Copy-KV-into-a-buffer baseline (pure torch): gather each sequence's blocks, then SDPA."""
    out = []
    for i in range(q.shape[0]):
        n = int(ctx_lens[i])
        j = torch.arange(n, device=q.device)
        slots = block_tables[i].long()[j // bs] * bs + j % bs
        k, v = kc[slots].transpose(0, 1)[None], vc[slots].transpose(0, 1)[None]
        out.append(F.scaled_dot_product_attention(q[i][:, None][None], k, v, enable_gqa=True)[0, :, 0])
    return torch.stack(out)
