"""Qwen3 dense forward pass over a flattened (varlen) batch of tokens and a paged KV cache."""
import json
from dataclasses import dataclass, field
import numpy as np
from .weights import load_dir, resolve


@dataclass
class SeqSlice:
    q_start: int          # row offset of this sequence's new tokens in the flat batch
    q_len: int            # number of new tokens this step
    ctx_len: int          # total KV length after this step (cached + new)
    block_table: list     # physical block ids


@dataclass
class Batch:
    tokens: list
    positions: list
    slot_mapping: list    # KV slot for each new token
    seqs: list            # [SeqSlice], decode (q_len==1) seqs first
    logits_rows: list = field(default_factory=list)  # rows that need logits


class Qwen3:
    def __init__(self, model, backend, quant=None):
        self.B = B = backend
        path = resolve(model)
        self.cfg = c = json.loads((path / "config.json").read_text())
        self.n_layers, self.H, self.Hkv = c["num_hidden_layers"], c["num_attention_heads"], c["num_key_value_heads"]
        self.D = c.get("head_dim") or c["hidden_size"] // self.H
        self.eps, self.theta = c["rms_norm_eps"], c["rope_theta"]
        raw = load_dir(path)

        def W(name):
            a, dt = raw.pop(name)
            return B.asarray(a, dt)

        self.embed = W("model.embed_tokens.weight")
        # some checkpoints store a full lm_head copy even when tied; reading it would cost bandwidth every token
        tied = c.get("tie_word_embeddings", "lm_head.weight" not in raw)
        self.lm_head = self.embed if tied else W("lm_head.weight")
        raw.pop("lm_head.weight", None)
        self.norm = W("model.norm.weight")
        self.layers = []
        for i in range(self.n_layers):
            p = f"model.layers.{i}."
            self.layers.append(dict(
                ln1=W(p + "input_layernorm.weight"), ln2=W(p + "post_attention_layernorm.weight"),
                qkv=B.concat([W(p + f"self_attn.{n}_proj.weight") for n in "qkv"]),
                o=W(p + "self_attn.o_proj.weight"),
                qn=W(p + "self_attn.q_norm.weight"), kn=W(p + "self_attn.k_norm.weight"),
                gu=B.concat([W(p + "mlp.gate_proj.weight"), W(p + "mlp.up_proj.weight")]),
                down=W(p + "mlp.down_proj.weight")))
        B.eval(self.embed, self.lm_head, self.norm, *[v for l in self.layers for v in l.values()])
        if quant == 4:  # every matmul weight, incl. the (tied) embedding / lm_head
            for L in self.layers:
                for k in ("qkv", "o", "gu", "down"):
                    L[k] = B.quantize(L[k])
            tied = self.lm_head is self.embed
            self.embed = B.quantize(self.embed)
            self.lm_head = self.embed if tied else B.quantize(self.lm_head)
        elif quant:
            raise ValueError(f"unsupported quant={quant}")
        self.kv_bytes_per_token = self.n_layers * 2 * self.Hkv * self.D * 2  # bf16

    def forward(self, batch: Batch, cache, block_size):
        """Returns logits [len(batch.logits_rows), V] (fp32)."""
        B, H, Hkv, D = self.B, self.H, self.Hkv, self.D
        T = len(batch.tokens)
        x = B.embedding(self.embed, B.asarray(np.asarray(batch.tokens, np.int32)))
        pos = B.asarray(np.asarray(batch.positions, np.int32))
        meta = B.prepare_attention(batch, block_size)
        h = B.rms_norm(x, self.layers[0]["ln1"], self.eps)
        for i, (L, layer_cache) in enumerate(zip(self.layers, cache)):
            a = B.attention_block(B.linear(h, L["qkv"]), L["qn"], L["kn"], layer_cache, meta,
                                  H, Hkv, D, self.eps, self.theta)
            h, x = B.add_rms_norm(x, B.linear(a, L["o"]), L["ln2"], self.eps)
            m = B.linear(B.silu_mul(B.linear(h, L["gu"])), L["down"])
            nxt = self.layers[i + 1]["ln1"] if i + 1 < self.n_layers else self.norm
            h, x = B.add_rms_norm(x, m, nxt, self.eps)
        rows = B.asarray(np.asarray(batch.logits_rows, np.int32))
        h = h[rows]
        if not batch.logits_rows:  # still a graph output: evaluating it runs the layers' KV writes
            return h
        return B.logits(h, self.lm_head)
