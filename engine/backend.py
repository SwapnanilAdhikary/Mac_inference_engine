"""All tensor math lives here. The model only calls these methods (plus +, reshape, slicing),
so a CUDA port replaces this file and nothing else."""
import numpy as np
import mlx.core as mx
from . import kernels as K

_DT = {"bf16": mx.bfloat16, "f16": mx.float16, "f32": mx.float32, "i32": mx.int32, "u32": mx.uint32}


class MLXBackend:
    name = "mlx"

    # ---- arrays -------------------------------------------------------------
    def asarray(self, a, dt=None):
        if dt == "bf16":  # numpy has no bf16: arrives as raw uint16 bits
            return mx.array(np.ascontiguousarray(a)).view(mx.bfloat16)
        return mx.array(a, dtype=_DT[dt]) if dt else mx.array(a)

    def concat(self, xs, axis=0):
        return mx.concatenate(xs, axis=axis)

    def zeros(self, shape, dt="bf16"):
        return mx.zeros(shape, dtype=_DT[dt])

    def eval(self, *xs):
        mx.eval(*xs)

    def tolist(self, x):
        return x.tolist()

    def free_gpu_bytes(self):
        return mx.device_info()["max_recommended_working_set_size"] - mx.get_active_memory()

    # ---- layers -------------------------------------------------------------
    def embedding(self, table, ids):
        return table[ids]

    def linear(self, x, w):  # w: [out, in] like HF
        return x @ w.T

    def logits(self, h, w):
        return (h @ w.T).astype(mx.float32)

    def rms_norm(self, x, w, eps):
        xf = x.astype(mx.float32)
        xf = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)
        return w * xf.astype(x.dtype)

    def add_rms_norm(self, x, delta, w, eps):
        """(norm(x + delta), x + delta)"""
        x = x + delta
        return self.rms_norm(x, w, eps), x

    def rope(self, x, positions, theta):
        """Non-interleaved (rotate-half) RoPE. x: [T, H, D], positions: [T] int32."""
        d = x.shape[-1]
        inv = theta ** (-mx.arange(0, d, 2, dtype=mx.float32) / d)           # [D/2]
        ang = positions.astype(mx.float32)[:, None] * inv[None, :]           # [T, D/2]
        cos, sin = mx.cos(ang)[:, None, :], mx.sin(ang)[:, None, :]
        xf = x.astype(mx.float32)
        x1, x2 = xf[..., : d // 2], xf[..., d // 2:]
        return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)

    def silu_mul(self, gu):  # gu: [T, 2I] = [gate | up]
        g, u = mx.split(gu, 2, axis=-1)
        return (g * mx.sigmoid(g.astype(mx.float32)).astype(g.dtype)) * u

    # ---- KV cache + attention ----------------------------------------------
    def alloc_kv(self, num_layers, num_slots, n_kv, head_dim, dt="bf16"):
        cache = [[mx.zeros((num_slots, n_kv, head_dim), dtype=_DT[dt]) for _ in range(2)]
                 for _ in range(num_layers)]
        mx.eval(cache)  # allocate now so an oversized cache fails at startup, not mid-request
        return cache

    def write_kv(self, cache, k, v, slots):
        cache[0][slots] = k
        cache[1][slots] = v

    def prepare_attention(self, batch, block_size):
        """Per-step metadata, shared by all layers. Decode rows are batched with padding;
        prefill chunks run one sequence at a time (the 'copy KV into one buffer' baseline)."""
        dec, pre = [], []
        for s in batch.seqs:
            (dec if s.q_len == 1 else pre).append(s)
        meta = {"slots": mx.array(np.asarray(batch.slot_mapping, np.int32)),
                "pos": mx.array(np.asarray(batch.positions, np.int32))}
        if dec:
            L = max(s.ctx_len for s in dec)
            slots = np.zeros((len(dec), L), np.int32)
            mask = np.full((len(dec), L), -np.inf, np.float32)
            for i, s in enumerate(dec):
                slots[i, :s.ctx_len] = _ctx_slots(s, block_size)
                mask[i, :s.ctx_len] = 0
            meta["dec"] = (mx.array([s.q_start for s in dec]), mx.array(slots), mx.array(mask))
        meta["pre"] = []
        for s in pre:
            pos_q = np.arange(s.ctx_len - s.q_len, s.ctx_len)[:, None]
            mask = np.where(np.arange(s.ctx_len)[None, :] <= pos_q, 0, -np.inf).astype(np.float32)
            meta["pre"].append((s.q_start, s.q_len, mx.array(_ctx_slots(s, block_size)), mx.array(mask)))
        meta["order"] = [s.q_start for s in dec] + [s.q_start for s in pre]
        return meta

    def attention_block(self, qkv, qn, kn, cache, meta, H, Hkv, D, eps, theta):
        """qkv [T, (H+2Hkv)*D] -> attention output [T, H*D]: QK-norm, RoPE, KV write, attention."""
        T = qkv.shape[0]
        q = self.rope(self.rms_norm(qkv[:, :H * D].reshape(T, H, D), qn, eps), meta["pos"], theta)
        k = self.rope(self.rms_norm(qkv[:, H * D:(H + Hkv) * D].reshape(T, Hkv, D), kn, eps), meta["pos"], theta)
        self.write_kv(cache, k, qkv[:, (H + Hkv) * D:].reshape(T, Hkv, D), meta["slots"])
        return self.attention(q, cache, meta, D ** -0.5).reshape(T, H * D)

    def attention(self, q, cache, meta, scale):
        """q: [T, H, D] -> [T, H, D]. Gathers each sequence's KV blocks into a dense buffer."""
        T, H, D = q.shape
        kc, vc = cache
        out = []
        if "dec" in meta:
            idx, slots, mask = meta["dec"]
            k, v = kc[slots], vc[slots]                                   # [S, L, Hkv, D]
            S, L, Hkv, _ = k.shape
            qd = q[idx].reshape(S, Hkv, H // Hkv, D)                      # [S, Hkv, G, D]
            s = (qd.astype(mx.float32) @ k.transpose(0, 2, 3, 1).astype(mx.float32)) * scale  # [S,Hkv,G,L]
            p = mx.softmax(s + mask[:, None, None, :], axis=-1)
            o = (p @ v.transpose(0, 2, 1, 3).astype(mx.float32))         # [S, Hkv, G, D]
            out.append(o.reshape(S, H, D).astype(q.dtype))
        for q0, ql, slots, mask in meta["pre"]:
            k, v = kc[slots], vc[slots]                                   # [L, Hkv, D]
            Hkv = k.shape[1]
            qp = q[q0:q0 + ql].reshape(ql, Hkv, H // Hkv, D).transpose(1, 2, 0, 3)  # [Hkv,G,Lq,D]
            s = (qp.astype(mx.float32) @ k.transpose(1, 2, 0)[:, None].astype(mx.float32)) * scale
            p = mx.softmax(s + mask, axis=-1)
            o = p @ v.transpose(1, 0, 2)[:, None].astype(mx.float32)     # [Hkv, G, Lq, D]
            out.append(o.transpose(2, 0, 1, 3).reshape(ql, H, D).astype(q.dtype))
        o = out[0] if len(out) == 1 else mx.concatenate(out, axis=0)
        # rows are grouped (decode first, then prefill); restore original token order
        if meta["order"] != sorted(meta["order"]):
            raise NotImplementedError("runner must place decode seqs before prefill seqs")
        return o

    # ---- sampling -----------------------------------------------------------
    def take_rows(self, x, rows):
        return x[mx.array(np.asarray(list(rows), np.int32))]

    def softmax_t(self, logits, temps):
        t = mx.array([max(t, 1e-5) for t in temps], dtype=mx.float32)[:, None]
        return mx.softmax(logits.astype(mx.float32) / t, axis=-1)

    def accept_ratios(self, p, q, d):
        """p: [k+1, V] target probs, q: k draft prob rows, d: k draft tokens -> [k] of p_i(d_i) / q_i(d_i)."""
        idx = mx.array(d)
        pd = p[mx.arange(len(d)), idx]
        qd = mx.stack([qi[t] for qi, t in zip(q, d)])
        return pd / mx.maximum(qd, 1e-20)

    def residual(self, p, q):
        r = mx.maximum(p - q, 0)
        return r / r.sum()

    def sample_probs(self, probs):
        probs = probs if probs.ndim == 2 else probs[None]
        return mx.random.categorical(mx.log(mx.maximum(probs, 1e-30)), axis=-1)

    def add_bias(self, logits, rows, toks, vals):
        return logits.at[mx.array(rows), mx.array(toks)].add(mx.array(vals, dtype=mx.float32))

    def sample(self, logits, temps, top_ps, top_ks, key=None):
        """logits: [S, V]. Per-row temperature/top-p/top-k; temp==0 means greedy."""
        logits = logits.astype(mx.float32)
        greedy = mx.argmax(logits, axis=-1)
        if all(t == 0 for t in temps):
            return greedy
        t = mx.array([max(t, 1e-5) for t in temps])[:, None]
        order = mx.argsort(-logits, axis=-1)
        sl = mx.take_along_axis(logits, order, axis=-1) / t
        p = mx.softmax(sl, axis=-1)
        cum = mx.cumsum(p, axis=-1) - p                                   # mass strictly before token
        rank = mx.arange(sl.shape[-1])[None, :]
        k = mx.array([k if k > 0 else sl.shape[-1] for k in top_ks])[:, None]
        keep = (cum < mx.array(top_ps)[:, None]) & (rank < k)
        pick = mx.random.categorical(mx.where(keep, sl, -mx.inf), key=key)
        sampled = mx.take_along_axis(order, pick[:, None], axis=-1)[:, 0]
        return mx.where(mx.array([t == 0 for t in temps]), greedy, sampled)


def _ctx_slots(s, block_size):
    j = np.arange(s.ctx_len)
    return (np.asarray(s.block_table, np.int32)[j // block_size] * block_size + j % block_size).astype(np.int32)


class Q4:
    """4-bit group-wise affine weight (see kernels.quantize4)."""
    def __init__(self, w):
        self.packed, self.scales, self.biases = K.quantize4(w)
        mx.eval(self.packed, self.scales, self.biases)


class MetalBackend(MLXBackend):
    """Same interface; norms, SiLU, QK-norm/RoPE/KV-write, paged attention and 4-bit matmuls run on our
    Metal kernels."""
    name = "metal"

    def quantize(self, w):
        return Q4(w)

    def linear(self, x, w):
        return K.qlinear(x, w.packed, w.scales, w.biases) if isinstance(w, Q4) else x @ w.T

    def logits(self, h, w):
        return self.linear(h, w).astype(mx.float32)

    def embedding(self, table, ids):
        if isinstance(table, Q4):
            return K.dequantize4(table.packed[ids], table.scales[ids], table.biases[ids])
        return table[ids]

    def __init__(self, split_part=512, split_min_ctx=2048, split_max_tgs=8):  # M5: 8 TGs already saturate DRAM
        self.split_part, self.split_min_ctx, self.split_max_tgs = split_part, split_min_ctx, split_max_tgs
        self._inv = {}

    def rms_norm(self, x, w, eps):
        return K.rms_norm(x, w, eps)

    def add_rms_norm(self, x, delta, w, eps):
        return K.rms_norm(x, w, eps, delta=delta)

    def silu_mul(self, gu):
        return K.silu_mul(gu)

    def prepare_attention(self, batch, block_size):
        seqs = batch.seqs
        dec = [s for s in seqs if s.q_len == 1]
        pre = [s for s in seqs if s.q_len > 1]
        assert seqs[:len(dec)] == dec, "runner must place decode seqs before prefill seqs"
        meta = {"slots": mx.array(np.asarray(batch.slot_mapping, np.int32)),
                "pos": mx.array(np.asarray(batch.positions, np.int32)), "bs": block_size,
                "n_dec": len(dec), "T": len(batch.tokens)}

        def tables(group):
            maxb = max(len(s.block_table) for s in group)
            bt = np.zeros((len(group), maxb), np.int32)
            for i, s in enumerate(group):
                bt[i, :len(s.block_table)] = s.block_table
            return mx.array(bt), mx.array(np.array([s.ctx_len for s in group], np.int32))
        if dec:
            bt, cl = tables(dec)
            max_ctx = max(s.ctx_len for s in dec)
            meta["dec"] = (mx.arange(len(dec), dtype=mx.int32), bt, cl, max_ctx)
        if pre:
            bt, cl = tables(pre)
            tiles, r = [], 0
            for i, s in enumerate(pre):
                p0 = s.ctx_len - s.q_len
                for j in range(0, s.q_len, K.BQ):
                    tiles.append([i, r + j, min(K.BQ, s.q_len - j), p0 + j])
                r += s.q_len
            meta["pre"] = (mx.array(np.array(tiles, np.int32)), len(tiles), bt, cl)
        return meta

    def attention_block(self, qkv, qn, kn, cache, meta, H, Hkv, D, eps, theta):
        if (theta, D) not in self._inv:
            self._inv[theta, D] = theta ** (-mx.arange(0, D, 2, dtype=mx.float32) / D)
        kc, vc = cache
        q, dep = K.qk_norm_rope_write(qkv, qn, kn, self._inv[theta, D], meta["pos"], meta["slots"], kc, vc,
                                      H, Hkv, eps)
        out, bs, S, T = [], meta["bs"], meta["n_dec"], meta["T"]
        if "dec" in meta:
            rows, bt, cl, max_ctx = meta["dec"]
            split = max_ctx >= self.split_min_ctx and S * Hkv < self.split_max_tgs
            out.append(K.paged_decode(q, rows, kc, vc, bt, cl, dep, H, Hkv, bs, max_ctx,
                                      part=self.split_part if split else None))
        if "pre" in meta:
            tiles, nt, bt, cl = meta["pre"]
            out.append(K.paged_prefill(q[S:], tiles, nt, kc, vc, bt, cl, dep, H, Hkv, bs, T - S))
        o = out[0] if len(out) == 1 else mx.concatenate(out, axis=0)
        return o.reshape(T, H * D)
