"""Paged decode attention in Triton. One program per (sequence, kv head): it loads the G = H/HKV query heads
that share the kv head once, walks the sequence's blocks through the block table (BS keys per iteration),
and keeps an fp32 online softmax. Same algorithm as the Metal kernel, with blocks instead of simdgroups."""
import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_kernel(q_ptr, k_ptr, v_ptr, bt_ptr, ctx_ptr, o_ptr, scale,
                         stride_bt, H: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, BS: tl.constexpr,
                         G: tl.constexpr, GP: tl.constexpr):
    s = tl.program_id(0)
    kvh = tl.program_id(1)
    g = tl.arange(0, GP)                      # query heads (padded to a power of two)
    d = tl.arange(0, D)
    j = tl.arange(0, BS)
    gmask = g < G
    q = tl.load(q_ptr + (s * H + kvh * G + g)[:, None] * D + d[None, :], mask=gmask[:, None], other=0.0)
    q = q.to(tl.float32) * scale
    ctx = tl.load(ctx_ptr + s)
    m = tl.full([GP], -float("inf"), tl.float32)
    l = tl.zeros([GP], tl.float32)
    acc = tl.zeros([GP, D], tl.float32)
    for b in range(0, tl.cdiv(ctx, BS)):
        blk = tl.load(bt_ptr + s * stride_bt + b)
        key = b * BS + j
        kmask = key < ctx
        off = ((blk * BS + j) * HKV + kvh)[:, None] * D + d[None, :]
        k = tl.load(k_ptr + off, mask=kmask[:, None], other=0.0).to(tl.float32)        # [BS, D]
        v = tl.load(v_ptr + off, mask=kmask[:, None], other=0.0).to(tl.float32)
        sc = tl.sum(q[:, None, :] * k[None, :, :], axis=2)                              # [GP, BS]
        sc = tl.where(kmask[None, :], sc, -float("inf"))
        m_new = tl.maximum(m, tl.max(sc, axis=1))
        corr = tl.exp(m - m_new)
        p = tl.exp(sc - m_new[:, None])
        l = l * corr + tl.sum(p, axis=1)
        acc = acc * corr[:, None] + tl.sum(p[:, :, None] * v[None, :, :], axis=1)
        m = m_new
    o = acc / l[:, None]
    tl.store(o_ptr + (s * H + kvh * G + g)[:, None] * D + d[None, :], o.to(tl.bfloat16), mask=gmask[:, None])


def paged_decode(q, kc, vc, block_tables, ctx_lens, bs):
    S, H, D = q.shape
    HKV = kc.shape[1]
    G = H // HKV
    o = torch.empty_like(q)
    _paged_decode_kernel[(S, HKV)](q.contiguous(), kc, vc, block_tables, ctx_lens, o, D ** -0.5,
                                   block_tables.stride(0), H=H, HKV=HKV, D=D, BS=bs, G=G,
                                   GP=max(2, triton.next_power_of_2(G)))
    return o
