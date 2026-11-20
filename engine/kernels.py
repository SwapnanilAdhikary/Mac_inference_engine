"""Hand-written Metal kernels (via mx.fast.metal_kernel, MLX's JIT hook for *our own* Metal source;
no mx.fast built-in ops are used here).

KV-cache layout: k_cache/v_cache are [num_slots, HKV, D] bf16, slot = block * BS + offset.
ponytail: metal_kernel has no input/output aliasing, so the KV write casts away const and writes
the cache buffer in place; it returns a 1-element `dep` array that every reader of the cache takes
as an input, which orders the graph and makes MLX insert a memory barrier. Upgrade path if MLX ever
reorders this: a C++ extension Primitive that donates the cache buffer to its output."""
import mlx.core as mx

BF16 = mx.bfloat16
_HDR = """
#include <metal_simdgroup_matrix>
using namespace metal;
"""
_HDR_NAX = """
#include <metal_simdgroup_matrix>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
"""

# ---------------------------------------------------------------- add + RMSNorm
# x_new = x + delta (ADD); out = w * bf16(x_new * rsqrt(mean(x_new^2) + eps)). One threadgroup per row.
_RMS_SRC = """
    uint row = threadgroup_position_in_grid.y, tid = thread_position_in_threadgroup.x;
    uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    threadgroup float part[32];
    float ss = 0.0f;
    for (uint i = tid; i < N; i += TG) {
        float v = float(x[row * N + i]);
#if ADD
        v = float(bfloat16_t(v + float(delta[row * N + i])));
        xo[row * N + i] = bfloat16_t(v);
#endif
        ss += v * v;
    }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) { ss = simd_sum(lane < TG / 32 ? part[lane] : 0.0f); if (lane == 0) part[0] = ss; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = metal::precise::rsqrt(part[0] / N + EPS);
    for (uint i = tid; i < N; i += TG) {
#if ADD
        float v = float(xo[row * N + i]);
#else
        float v = float(x[row * N + i]);
#endif
        out[row * N + i] = bfloat16_t(float(w[i]) * float(bfloat16_t(v * inv)));
    }
"""
_rms_cache = {}


def rms_norm(x, w, eps, delta=None, tg=256):
    """norm(x) or, with delta, (norm(x + delta), x + delta) in one pass."""
    add = delta is not None
    if (eps, add) not in _rms_cache:
        _rms_cache[eps, add] = mx.fast.metal_kernel(
            name="add_rms_norm" if add else "rms_norm", source=_RMS_SRC,
            header=_HDR + f"#define ADD {int(add)}\n#define EPS {eps:.10e}f\n",
            input_names=["x", "delta", "w"] if add else ["x", "w"], output_names=["out", "xo"] if add else ["out"])
    shape, N = x.shape, x.shape[-1]
    R = x.size // N
    ins = [x.reshape(R, N)] + ([delta.reshape(R, N)] if add else []) + [w]
    outs = _rms_cache[eps, add](inputs=ins, template=[("N", N), ("TG", tg)], grid=(tg, R, 1),
                                threadgroup=(tg, 1, 1), output_shapes=[(R, N)] * (1 + add),
                                output_dtypes=[x.dtype] * (1 + add))
    return (outs[0].reshape(shape), outs[1].reshape(shape)) if add else outs[0].reshape(shape)


# ---------------------------------------------------------------- SiLU(gate) * up
_silu_k = mx.fast.metal_kernel(name="silu_mul", input_names=["gu"], output_names=["out"], header=_HDR, source="""
    uint i = thread_position_in_grid.x, t = thread_position_in_grid.y;
    if (i >= I) return;
    float g = float(gu[t * 2 * I + i]);
    float s = float(bfloat16_t(g / (1.0f + metal::precise::exp(-g))));
    out[t * I + i] = bfloat16_t(s * float(gu[t * 2 * I + I + i]));
""")


def silu_mul(gu):
    T, I2 = gu.shape
    I = I2 // 2
    return _silu_k(inputs=[gu], template=[("I", I)], grid=(-(-I // 256) * 256, T, 1), threadgroup=(256, 1, 1),
                   output_shapes=[(T, I)], output_dtypes=[gu.dtype])[0]


# ---------------------------------------------------------------- QK-norm -> RoPE -> KV-cache write (fused)
# One simdgroup per (token, head); D == 128 -> 4 elements per lane. Heads [0,H) are q heads,
# [H, H+HKV) are k heads (which also copy the matching v head). RoPE partner (e +- 64) lives 16 lanes away.
_QKR_SRC = """
    uint lane = thread_index_in_simdgroup;
    uint h = thread_position_in_grid.y, t = thread_position_in_grid.z;
    const uint QKV = (H + 2 * HKV) * D;
    bool is_q = h < H;
    uint src = t * QKV + h * D + lane * 4;
    float x[4], ss = 0.0f;
    for (int j = 0; j < 4; j++) { x[j] = float(qkv[src + j]); ss += x[j] * x[j]; }
    float inv = metal::precise::rsqrt(simd_sum(ss) / D + EPS);
    float p = float(pos[t]);
    float o[4];
    for (int j = 0; j < 4; j++) {
        float w = float(is_q ? qn[lane * 4 + j] : kn[lane * 4 + j]);
        float xn = float(bfloat16_t(w * float(bfloat16_t(x[j] * inv))));
        float partner = simd_shuffle_xor(xn, 16);
        float ang = p * inv_freq[(lane * 4 + j) % (D / 2)];
        float c = metal::precise::cos(ang), s = metal::precise::sin(ang);
        o[j] = lane < 16 ? xn * c - partner * s : xn * c + partner * s;
    }
    if (is_q) {
        uint dst = t * H * D + h * D + lane * 4;
        for (int j = 0; j < 4; j++) q_out[dst + j] = bfloat16_t(o[j]);
    } else {
        uint kh = h - H;
        uint dst = (uint(slots[t]) * HKV + kh) * D + lane * 4;
        device bfloat16_t* kc = const_cast<device bfloat16_t*>(k_cache);
        device bfloat16_t* vc = const_cast<device bfloat16_t*>(v_cache);
        uint vsrc = t * QKV + (H + HKV + kh) * D + lane * 4;
        for (int j = 0; j < 4; j++) { kc[dst + j] = bfloat16_t(o[j]); vc[dst + j] = qkv[vsrc + j]; }
    }
    if (t == 0 && h == 0 && lane == 0) dep[0] = 1;
"""
_qkr_cache = {}


def qk_norm_rope_write(qkv, qn, kn, inv_freq, pos, slots, k_cache, v_cache, H, HKV, eps):
    """qkv [T, (H+2HKV)*128] -> q [T, H, 128] (normed + roped); k (normed + roped) and v are written
    into the cache at `slots`. Returns (q, dep)."""
    T, D = qkv.shape[0], 128
    assert qkv.shape[1] == (H + 2 * HKV) * D
    if eps not in _qkr_cache:
        _qkr_cache[eps] = mx.fast.metal_kernel(
            name="qk_norm_rope_write", header=_HDR + f"#define EPS {eps:.10e}f\n", source=_QKR_SRC,
            input_names=["qkv", "qn", "kn", "inv_freq", "pos", "slots", "k_cache", "v_cache"],
            output_names=["q_out", "dep"])
    q, dep = _qkr_cache[eps](inputs=[qkv, qn, kn, inv_freq, pos, slots, k_cache, v_cache],
                             template=[("H", H), ("HKV", HKV), ("D", D)],
                             grid=(32, H + HKV, T), threadgroup=(32, 1, 1),
                             output_shapes=[(T, H, D), (1,)], output_dtypes=[qkv.dtype, mx.int32])
    return q, dep


# ---------------------------------------------------------------- paged decode attention
# Threadgroup = (kv head, seq[, partition]); SIMDS simdgroups stride over keys; lanes split D (4 each);
# all G = H/HKV query heads sharing the kv head are handled together (K/V read once). fp32 online softmax.
# SPLIT: each partition of PART keys writes (unnormalized acc, max, sum) for a separate reduce pass.
_DEC_SRC = """
    const uint G = H / HKV;
    uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    uint tid = thread_position_in_threadgroup.x;
    uint kvh = threadgroup_position_in_grid.x, s = threadgroup_position_in_grid.y;
    uint part = threadgroup_position_in_grid.z, nparts = threadgroups_per_grid.z;
    int ctx = ctx_lens[s];
#if SPLIT
    int k0 = int(part) * PART, k1 = min(ctx, k0 + PART);
#else
    int k0 = 0, k1 = ctx;
#endif
    const float scale = 1.0f / sqrt(float(D));
    float q[G][4], acc[G][4], m[G], l[G];
    for (uint g = 0; g < G; g++) {
        uint qb = (uint(q_rows[s]) * H + kvh * G + g) * D + lane * 4;
        for (int j = 0; j < 4; j++) { q[g][j] = float(qs[qb + j]) * scale; acc[g][j] = 0.0f; }
        m[g] = -INFINITY; l[g] = 0.0f;
    }
    auto bt = block_tables + s * block_tables_shape[1];   // runtime: no recompile per width
    for (int k = k0 + int(sg); k < k1; k += SIMDS) {
        uint slot = uint(bt[k / BS]) * BS + uint(k % BS);
        uint kb = (slot * HKV + kvh) * D + lane * 4;
        float kv[4], vv[4];
        for (int j = 0; j < 4; j++) { kv[j] = float(k_cache[kb + j]); vv[j] = float(v_cache[kb + j]); }
        for (uint g = 0; g < G; g++) {
            float sc = simd_sum(q[g][0] * kv[0] + q[g][1] * kv[1] + q[g][2] * kv[2] + q[g][3] * kv[3]);
            float mn = max(m[g], sc);
            float corr = metal::precise::exp(m[g] - mn), e = metal::precise::exp(sc - mn);
            l[g] = l[g] * corr + e;
            for (int j = 0; j < 4; j++) acc[g][j] = acc[g][j] * corr + e * vv[j];
            m[g] = mn;
        }
    }
    // combine simdgroups through threadgroup memory
    threadgroup float tm[SIMDS][G], tl[SIMDS][G], tacc[SIMDS][G][D];
    for (uint g = 0; g < G; g++) {
        if (lane == 0) { tm[sg][g] = m[g]; tl[sg][g] = l[g]; }
        for (int j = 0; j < 4; j++) tacc[sg][g][lane * 4 + j] = acc[g][j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < G * D; i += SIMDS * 32) {
        uint g = i / D, d = i % D;
        float M = -INFINITY;
        for (int w = 0; w < SIMDS; w++) M = max(M, tm[w][g]);
        float L = 0.0f, A = 0.0f;
        for (int w = 0; w < SIMDS; w++) {
            float f = tm[w][g] == -INFINITY ? 0.0f : metal::precise::exp(tm[w][g] - M);
            L += tl[w][g] * f; A += tacc[w][g][d] * f;
        }
        uint hq = kvh * G + g;
#if SPLIT
        uint pi = (s * H + hq) * nparts + part;
        o_part[pi * D + d] = A;
        if (d == 0) { m_part[pi] = M; l_part[pi] = L; }
#else
        out[(s * H + hq) * D + d] = bfloat16_t(L > 0.0f ? A / L : 0.0f);
#endif
    }
"""
_dec_kernels = {}


def _dec_kernel(split):
    if split not in _dec_kernels:
        _dec_kernels[split] = mx.fast.metal_kernel(
            name="paged_decode_split" if split else "paged_decode", source=_DEC_SRC,
            header=_HDR + f"#define SPLIT {int(split)}\n",
            input_names=["qs", "q_rows", "k_cache", "v_cache", "block_tables", "ctx_lens", "dep"],
            output_names=["o_part", "m_part", "l_part"] if split else ["out"])
    return _dec_kernels[split]


_red_k = mx.fast.metal_kernel(name="paged_decode_reduce", input_names=["o_part", "m_part", "l_part"],
                              output_names=["out"], header=_HDR, source="""
    uint d = thread_position_in_grid.x, sh = thread_position_in_grid.y;   // sh = seq * H + head
    float M = -INFINITY;
    for (uint p = 0; p < P; p++) M = max(M, m_part[sh * P + p]);
    float L = 0.0f, A = 0.0f;
    for (uint p = 0; p < P; p++) {
        float mp = m_part[sh * P + p];
        float f = mp == -INFINITY ? 0.0f : metal::precise::exp(mp - M);
        L += l_part[sh * P + p] * f; A += o_part[(sh * P + p) * D + d] * f;
    }
    out[sh * D + d] = bfloat16_t(L > 0.0f ? A / L : 0.0f);
""")


def paged_decode(q, q_rows, k_cache, v_cache, block_tables, ctx_lens, dep, H, HKV, BS,
                 max_ctx, part=None, simds=8):
    """q: [T, H, D]; q_rows [S] picks each sequence's query row. Returns [S, H, D].
    part=None: one pass. part=N: split each sequence into N-key partitions + reduce (long contexts)."""
    S, D = q_rows.shape[0], q.shape[-1]
    # only values fixed for a model are template params: each new template value is a ~40 ms JIT compile
    tpl = [("H", H), ("HKV", HKV), ("D", D), ("BS", BS), ("SIMDS", simds)]
    ins = [q, q_rows, k_cache, v_cache, block_tables, ctx_lens, dep]
    if part is None:
        return _dec_kernel(False)(inputs=ins, template=tpl + [("PART", 0)], grid=(simds * 32 * HKV, S, 1),
                                  threadgroup=(simds * 32, 1, 1), output_shapes=[(S, H, D)],
                                  output_dtypes=[q.dtype])[0]
    P = -(-max_ctx // part)
    o_p, m_p, l_p = _dec_kernel(True)(inputs=ins, template=tpl + [("PART", part)],
                                      grid=(simds * 32 * HKV, S, P), threadgroup=(simds * 32, 1, 1),
                                      output_shapes=[(S * H * P, D), (S * H * P,), (S * H * P,)],
                                      output_dtypes=[mx.float32] * 3)
    return _red_k(inputs=[o_p, m_p, l_p], template=[("P", P), ("D", D)], grid=(D, S * H, 1),
                  threadgroup=(D, 1, 1), output_shapes=[(S, H, D)], output_dtypes=[q.dtype])[0]


# ---------------------------------------------------------------- prefill: tiled causal flash attention
# Threadgroup = (query tile of BQ = 8*SGH rows, kv head). It serves all G = H/HKV query heads that share the
# kv head (SGH simdgroups per head, 8 query rows each), so every K/V tile gathered from the paged cache is
# reused G*SGH times. S = QK^T and O += PV use 8x8 simdgroup MMA with fp32 accumulators; online softmax in
# base 2 (log2(e) folded into the q scale). Apple 8x8 fragment layout: lane holds row fm, cols fn, fn+1.
_PRE_SRC = """
    const uint G = H / HKV;
    uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    uint tid = thread_position_in_threadgroup.x, nthreads = threads_per_threadgroup.x;
    uint tile = threadgroup_position_in_grid.x, kvh = threadgroup_position_in_grid.y;
    uint h = kvh * G + sg / SGH;                           // this simdgroup's q head
    int seq = tiles[tile * 4 + 0], row0 = tiles[tile * 4 + 1], nrows = tiles[tile * 4 + 2], pos0 = tiles[tile * 4 + 3];
    int ctx = ctx_lens[seq];
    auto bt = block_tables + seq * block_tables_shape[1];
    const float scale = 1.4426950408889634f / sqrt(float(D));   // log2(e) / sqrt(D)

    short qid = lane / 4;
    short fm = (qid & 4) + ((lane / 2) % 4);
    short fn = (qid & 2) * 2 + (lane % 2) * 2;
    int my_row = int(sg % SGH) * 8 + fm;                   // row inside the tile
    int qpos = pos0 + my_row;

    simdgroup_matrix<float, 8, 8> Qf[D / 8];
    int qr = min(my_row, nrows - 1);
    for (uint kk = 0; kk < D / 8; kk++) {
        uint b = (uint(row0 + qr) * H + h) * D + kk * 8 + fn;
        Qf[kk].thread_elements()[0] = float(qs[b]) * scale;
        Qf[kk].thread_elements()[1] = float(qs[b + 1]) * scale;
    }
    simdgroup_matrix<float, 8, 8> Of[D / 8];
    for (uint j = 0; j < D / 8; j++) Of[j] = simdgroup_matrix<float, 8, 8>(0.0f);
    float m_i = -INFINITY, l_i = 0.0f;

    threadgroup bfloat16_t Ks[BK][D + 8];
    threadgroup bfloat16_t Vs[BK][D + 8];
    int kend = min(ctx, pos0 + nrows);                     // causal: keys up to the tile's last query
    int my_kend = min(ctx, qpos + 1);                      // ... and this row's own limit
    for (int kt = 0; kt < kend; kt += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < BK * D / 8; i += nthreads) {   // paged gather, 16 bytes per thread per op
            uint r = i / (D / 8), c = (i % (D / 8)) * 8;
            int key = kt + int(r);
            uint4 kv = uint4(0), vv = uint4(0);
            if (key < ctx) {
                uint slot = uint(bt[key / BS]) * BS + uint(key % BS);
                uint b = (slot * HKV + kvh) * D + c;
                kv = *(const device uint4*)(k_cache + b);
                vv = *(const device uint4*)(v_cache + b);
            }
            *(threadgroup uint4*)(&Ks[r][c]) = kv;
            *(threadgroup uint4*)(&Vs[r][c]) = vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<float, 8, 8> Sf[BK / 8];
        for (uint n = 0; n < BK / 8; n++) {
            Sf[n] = simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint kk = 0; kk < D / 8; kk++) {
                simdgroup_matrix<float, 8, 8> Kt;           // Kt[r=dim][c=key] = K[key][dim]
                Kt.thread_elements()[0] = float(Ks[n * 8 + fn][kk * 8 + fm]);
                Kt.thread_elements()[1] = float(Ks[n * 8 + fn + 1][kk * 8 + fm]);
                simdgroup_multiply_accumulate(Sf[n], Qf[kk], Kt, Sf[n]);
            }
        }
        float rmax = -INFINITY;
        bool full = kt + BK <= my_kend;                    // whole tile visible to this row: skip masking
        for (uint n = 0; n < BK / 8; n++) {
            for (int e = 0; e < 2; e++) {
                float v = Sf[n].thread_elements()[e];
                if (!full && kt + int(n * 8 + fn + e) >= my_kend) v = -INFINITY;
                Sf[n].thread_elements()[e] = v;
                rmax = max(rmax, v);
            }
        }
        rmax = max(rmax, simd_shuffle_xor(rmax, 1));
        rmax = max(rmax, simd_shuffle_xor(rmax, 8));
        float m_new = max(m_i, rmax);
        float corr = m_new == -INFINITY ? 1.0f : fast::exp2(m_i - m_new);
        float rsum = 0.0f;
        for (uint n = 0; n < BK / 8; n++) {
            for (int e = 0; e < 2; e++) {
                float v = Sf[n].thread_elements()[e];
                float pexp = v == -INFINITY ? 0.0f : fast::exp2(v - m_new);
                Sf[n].thread_elements()[e] = pexp;
                rsum += pexp;
            }
        }
        rsum += simd_shuffle_xor(rsum, 1);
        rsum += simd_shuffle_xor(rsum, 8);
        l_i = l_i * corr + rsum;
        m_i = m_new;
        for (uint j = 0; j < D / 8; j++) {
            Of[j].thread_elements()[0] *= corr;
            Of[j].thread_elements()[1] *= corr;
        }
        for (uint j = 0; j < D / 8; j++) {
            for (uint n = 0; n < BK / 8; n++) {
                simdgroup_matrix<float, 8, 8> Vf;
                Vf.thread_elements()[0] = float(Vs[n * 8 + fm][j * 8 + fn]);
                Vf.thread_elements()[1] = float(Vs[n * 8 + fm][j * 8 + fn + 1]);
                simdgroup_multiply_accumulate(Of[j], Sf[n], Vf, Of[j]);
            }
        }
    }
    if (my_row < nrows) {
        float inv_l = l_i > 0.0f ? 1.0f / l_i : 0.0f;
        uint b = (uint(row0 + my_row) * H + h) * D;
        for (uint j = 0; j < D / 8; j++) {
            out[b + j * 8 + fn] = bfloat16_t(Of[j].thread_elements()[0] * inv_l);
            out[b + j * 8 + fn + 1] = bfloat16_t(Of[j].thread_elements()[1] * inv_l);
        }
    }
"""
_pre_k = mx.fast.metal_kernel(name="paged_prefill", header=_HDR, source=_PRE_SRC,
                              input_names=["qs", "tiles", "k_cache", "v_cache", "block_tables", "ctx_lens", "dep"],
                              output_names=["out"])

# ---------------------------------------------------------------- prefill on the M5 neural accelerators
# Same tiling idea, but the two matmuls run as Metal 4 tensor ops (mpp::tensor_ops::matmul2d 16x32x16,
# bf16 in, fp32 accumulate) on register-resident cooperative tensors. Each simdgroup owns 16 query rows
# of one q head; a lane holds rows fm, fm+8 and 4 consecutive columns from fn of every 16x16 fragment.
# The S (destination) fragment layout equals the P (left operand) layout, so softmax needs no shuffles
# beyond the row reductions (lanes sharing a row differ in lane bits 0 and 3).
_NAX_SRC = """
    const uint G = H / HKV;
    using namespace mpp::tensor_ops;
    ushort lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint tid = thread_position_in_threadgroup.x, nthreads = threads_per_threadgroup.x;
    uint tile = threadgroup_position_in_grid.x, kvh = threadgroup_position_in_grid.y;
    uint h = kvh * G + sg / SGH;
    int seq = tiles[tile * 4 + 0], row0 = tiles[tile * 4 + 1], nrows = tiles[tile * 4 + 2], pos0 = tiles[tile * 4 + 3];
    int ctx = ctx_lens[seq];
    auto bt = block_tables + seq * block_tables_shape[1];
    const float scale = 1.4426950408889634f / sqrt(float(D));   // log2(e) / sqrt(D), applied to S in fp32

    short qid = lane >> 2;
    short fm = (qid & 4) | ((lane >> 1) & 3);
    short fn = ((qid & 2) | (lane & 1)) * 4;
    int rbase = int(sg % SGH) * 16;
    int qpos[2] = {pos0 + rbase + fm, pos0 + rbase + fm + 8};

    bfloat qf[D / 16][8];
    for (short kk = 0; kk < D / 16; kk++)
        for (short i = 0; i < 2; i++) {
            int r = min(rbase + fm + 8 * i, nrows - 1);
            uint b = (uint(row0 + r) * H + h) * D + kk * 16 + fn;
            for (short j = 0; j < 4; j++) qf[kk][i * 4 + j] = qs[b + j];
        }
    float o[D / 32][16];
    for (short nb = 0; nb < D / 32; nb++) for (short e = 0; e < 16; e++) o[nb][e] = 0.0f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};

    constexpr auto dS = matmul2d_descriptor(16, 32, 16, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto dO = matmul2d_descriptor(16, 32, 16, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dS, execution_simdgroup> opS;
    matmul2d<dO, execution_simdgroup> opO;

    threadgroup bfloat Ks[32][D + 8];
    threadgroup bfloat Vs[32][D + 8];
    int kend = min(ctx, pos0 + nrows);
    int sg_last = min(pos0 + rbase + 15, pos0 + nrows - 1);     // last query position this simdgroup needs
    for (int kt = 0; kt < kend; kt += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 32 * D / 8; i += nthreads) {
            uint r = i / (D / 8), c = (i % (D / 8)) * 8;
            int key = kt + int(r);
            uint4 kv = uint4(0), vv = uint4(0);
            if (key < ctx) {
                uint slot = uint(bt[key / BS]) * BS + uint(key % BS);
                uint b = (slot * HKV + kvh) * D + c;
                kv = *(const device uint4*)(k_cache + b);
                vv = *(const device uint4*)(v_cache + b);
            }
            *(threadgroup uint4*)(&Ks[r][c]) = kv;
            *(threadgroup uint4*)(&Vs[r][c]) = vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kt > sg_last) continue;                              // fully masked for this simdgroup's rows

        auto ca = opS.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
        auto cb = opS.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
        auto cs = opS.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
        for (short e = 0; e < 16; e++) cs[e] = 0.0f;
        for (short kk = 0; kk < D / 16; kk++) {
            for (short e = 0; e < 8; e++) ca[e] = qf[kk][e];
            for (short n = 0; n < 2; n++)
                for (short i = 0; i < 2; i++)
                    for (short j = 0; j < 4; j++) cb[n * 8 + i * 4 + j] = Ks[n * 16 + fm + 8 * i][kk * 16 + fn + j];
            opS.run(ca, cb, cs);
        }
        float p[16], rmax[2] = {-INFINITY, -INFINITY};
        for (short e = 0; e < 16; e++) {
            short n = e / 8, i = (e % 8) / 4, j = e % 4;
            int key = kt + n * 16 + fn + j;
            float v = (key <= qpos[i] && key < ctx) ? cs[e] * scale : -INFINITY;
            p[e] = v;
            rmax[i] = max(rmax[i], v);
        }
        float corr[2];
        for (short i = 0; i < 2; i++) {
            rmax[i] = max(rmax[i], simd_shuffle_xor(rmax[i], 1));
            rmax[i] = max(rmax[i], simd_shuffle_xor(rmax[i], 8));
            float mn = max(m[i], rmax[i]);
            corr[i] = mn == -INFINITY ? 1.0f : fast::exp2(m[i] - mn);
            m[i] = mn;
        }
        float rs[2] = {0.0f, 0.0f};
        for (short e = 0; e < 16; e++) {
            short i = (e % 8) / 4;
            p[e] = p[e] == -INFINITY ? 0.0f : fast::exp2(p[e] - m[i]);
            rs[i] += p[e];
        }
        for (short i = 0; i < 2; i++) {
            rs[i] += simd_shuffle_xor(rs[i], 1);
            rs[i] += simd_shuffle_xor(rs[i], 8);
            l[i] = l[i] * corr[i] + rs[i];
        }
        auto pa = opO.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
        auto vb = opO.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
        auto co = opO.template get_destination_cooperative_tensor<decltype(pa), decltype(vb), float>();
        for (short nb = 0; nb < D / 32; nb++) {
            for (short e = 0; e < 16; e++) co[e] = o[nb][e] * corr[(e % 8) / 4];
            for (short ks = 0; ks < 2; ks++) {
                for (short e = 0; e < 8; e++) pa[e] = bfloat(p[ks * 8 + e]);
                for (short n = 0; n < 2; n++)
                    for (short i = 0; i < 2; i++)
                        for (short j = 0; j < 4; j++)
                            vb[n * 8 + i * 4 + j] = Vs[ks * 16 + fm + 8 * i][nb * 32 + n * 16 + fn + j];
                opO.run(pa, vb, co);
            }
            for (short e = 0; e < 16; e++) o[nb][e] = co[e];
        }
    }
    for (short i = 0; i < 2; i++) {
        int r = rbase + fm + 8 * i;
        if (r >= nrows) continue;
        float inv_l = l[i] > 0.0f ? 1.0f / l[i] : 0.0f;
        uint b = (uint(row0 + r) * H + h) * D;
        for (short nb = 0; nb < D / 32; nb++)
            for (short n = 0; n < 2; n++)
                for (short j = 0; j < 4; j++)
                    out[b + nb * 32 + n * 16 + fn + j] = bfloat(o[nb][n * 8 + i * 4 + j] * inv_l);
    }
"""
_nax_k = mx.fast.metal_kernel(name="paged_prefill_nax", header=_HDR_NAX, source=_NAX_SRC,
                              input_names=["qs", "tiles", "k_cache", "v_cache", "block_tables", "ctx_lens", "dep"],
                              output_names=["out"])
NAX_SGH = 2      # simdgroups per q head -> 32 query rows per tile


def has_nax():
    """M5-class GPUs (Apple10+) have the per-core neural accelerators that Metal 4 tensor ops use."""
    arch = mx.device_info().get("architecture", "")
    try:
        return int("".join(c for c in arch.split("_")[1] if c.isdigit())) >= 17
    except (IndexError, ValueError):
        return False


SGH = 4          # simdgroup-MMA fallback: simdgroups per q head -> BQ = 8 * SGH query rows per tile
BQ = 32          # rows per prefill tile (both kernels use 32)


def paged_prefill(q, tiles, num_tiles, k_cache, v_cache, block_tables, ctx_lens, dep, H, HKV, BS, T_out, nax=None):
    """q: [T, H, D]. tiles: [num_tiles, 4] int32 rows of (seq, first q row, rows in tile, pos of first row).
    Returns [T_out, H, D]; rows not covered by any tile are 0."""
    D, G = q.shape[-1], H // HKV
    nax = has_nax() if nax is None else nax
    kern, sgh = (_nax_k, NAX_SGH) if nax else (_pre_k, SGH)
    assert (16 if nax else 8) * sgh == BQ
    return kern(inputs=[q, tiles, k_cache, v_cache, block_tables, ctx_lens, dep],
                template=[("H", H), ("HKV", HKV), ("D", D), ("BS", BS), ("BK", 32), ("SGH", sgh)],
                grid=(32 * sgh * G * num_tiles, HKV, 1), threadgroup=(32 * sgh * G, 1, 1),
                output_shapes=[(T_out, H, D)], output_dtypes=[q.dtype], init_value=0)[0]


# ---------------------------------------------------------------- 4-bit group-wise affine weights
# w[n, k] ~= scale[n, k/64] * q + bias[n, k/64], q in [0, 15]; 8 nibbles per uint32 (element k+j = nibble j).
GROUP = 64
_QMV_SRC = """
    // out[m, n] = sum_k x[m, k] * w[n, k]. Simdgroup = ROWS output rows; lane covers 8 consecutive k per step.
    uint lane = thread_index_in_simdgroup;
    uint n0 = (threadgroup_position_in_grid.x * SIMDS + simdgroup_index_in_threadgroup) * ROWS;
    float acc[ROWS][M];
    for (int r = 0; r < ROWS; r++) for (int m = 0; m < M; m++) acc[r][m] = 0.0f;
    for (uint k = lane * 8; k < K; k += 256) {
        float xv[M][8], xs[M];
        for (int m = 0; m < M; m++) {
            xs[m] = 0.0f;
            for (int j = 0; j < 8; j++) { xv[m][j] = float(x[m * K + k + j]); xs[m] += xv[m][j]; }
        }
        for (int r = 0; r < ROWS; r++) {
            uint n = n0 + r;
            if (n >= N) break;
            uint32_t wq = w[n * (K / 8) + k / 8];
            float s = float(scales[n * (K / 64) + k / 64]), b = float(biases[n * (K / 64) + k / 64]);
            float q[8];
            for (int j = 0; j < 8; j++) q[j] = float((wq >> (4 * j)) & 0xF);
            for (int m = 0; m < M; m++) {
                float d = 0.0f;
                for (int j = 0; j < 8; j++) d += q[j] * xv[m][j];
                acc[r][m] += s * d + b * xs[m];
            }
        }
    }
    for (int r = 0; r < ROWS; r++) {
        uint n = n0 + r;
        if (n >= N) break;
        for (int m = 0; m < M; m++) {
            float v = simd_sum(acc[r][m]);
            if (lane == 0) out[m * N + n] = bfloat16_t(v);
        }
    }
"""
_qmv_k = mx.fast.metal_kernel(name="qmv4", header=_HDR, source=_QMV_SRC,
                              input_names=["x", "w", "scales", "biases"], output_names=["out"])
_DQ_SRC = """
    uint i = thread_position_in_grid.x, n = thread_position_in_grid.y;   // i indexes uint32 words in row n
    if (i >= K / 8) return;
    uint32_t wq = w[n * (K / 8) + i];
    float s = float(scales[n * (K / 64) + i / 8]), b = float(biases[n * (K / 64) + i / 8]);
    for (int j = 0; j < 8; j++) out[n * K + i * 8 + j] = bfloat16_t(s * float((wq >> (4 * j)) & 0xF) + b);
"""
_dq_k = mx.fast.metal_kernel(name="dequant4", header=_HDR, source=_DQ_SRC,
                             input_names=["w", "scales", "biases"], output_names=["out"])
QMV_MAX_M = 4      # crossovers measured on M5 (bench/kernels.py --q4)
QMM_MAX_M = 96


def quantize4(w):
    """bf16 [N, K] -> (packed uint32 [N, K/8], scales bf16 [N, K/64], biases bf16 [N, K/64]). Built from basic ops.
    Zero-exact affine: the scale is snapped so the larger-magnitude group edge is the bias and 0 is a grid
    point. Plain min/max scaling leaves 0 between levels, so in outlier-heavy groups every near-zero weight
    gets the same up-to-half-step error, which adds up coherently (measured: ppl 6.24 -> 8.84 vs 6.39 here)."""
    N, Kd = w.shape
    assert Kd % 256 == 0, "K must be a multiple of 256 (32 lanes x 8 nibbles)"
    g = w.astype(mx.float32).reshape(N, Kd // GROUP, GROUP)
    lo, hi = g.min(axis=-1, keepdims=True), g.max(axis=-1, keepdims=True)
    lo_edge = mx.abs(lo) > mx.abs(hi)
    step = mx.maximum((hi - lo) / 15, 1e-8)
    signed = mx.where(lo_edge, step, -step)              # negative scale: count down from the max edge
    edge = mx.where(lo_edge, lo, hi)
    q0 = mx.round(edge / signed)                         # zero sits at level -q0
    scale = mx.where(q0 != 0, edge / mx.where(q0 != 0, q0, 1), signed)
    bias = mx.where(q0 == 0, 0, edge)
    s16, b16 = scale.astype(mx.bfloat16), bias.astype(mx.bfloat16)
    # quantize against the bf16-rounded scale/bias that dequantization will actually use
    q = mx.clip(mx.round((g - b16.astype(mx.float32)) / s16.astype(mx.float32)), 0, 15).astype(mx.uint32)
    q = q.reshape(N, Kd // 8, 8) << (mx.arange(8, dtype=mx.uint32) * 4)
    packed = q.sum(axis=-1).astype(mx.uint32)
    return packed, s16.reshape(N, -1), b16.reshape(N, -1)


def dequantize4(packed, scales, biases):
    N, K = packed.shape[0], packed.shape[1] * 8
    return _dq_k(inputs=[packed, scales, biases], template=[("K", K)], grid=(-(-(K // 8) // 64) * 64, N, 1),
                 threadgroup=(64, 1, 1), output_shapes=[(N, K)], output_dtypes=[mx.bfloat16])[0]


_QMM_SRC = """
    // out[M, N] = x[M, K] @ W[N, K]^T with 4-bit W: weight tiles are unpacked into threadgroup memory as bf16,
    // then multiplied on the tensor units (matmul2d 16x32x16, fp32 accumulate). Tile = BM x BN, BM = 16*WM,
    // BN = 32*WN; simdgroup (wm, wn) owns a 16 x 32 output block. Weights are read from DRAM once per M-tile.
    using namespace mpp::tensor_ops;
    const uint M = x_shape[0];   // runtime: decode batches vary every step
    const uint BM = 16 * WM, BN = 32 * WN, BK = 64;
    ushort lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup, tid = thread_position_in_threadgroup.x;
    uint nthreads = threads_per_threadgroup.x;
    uint n0 = threadgroup_position_in_grid.x * BN, m0 = threadgroup_position_in_grid.y * BM;
    uint wm = sg / WN, wn = sg % WN;
    short qid = lane >> 2;
    short fm = (qid & 4) | ((lane >> 1) & 3);
    short fn = ((qid & 2) | (lane & 1)) * 4;

    threadgroup bfloat Xs[BM][BK + 8];
    threadgroup bfloat Ws[BN][BK + 8];
    constexpr auto desc = matmul2d_descriptor(16, 32, 16, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto cb = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto cc = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    for (short e = 0; e < 16; e++) cc[e] = 0.0f;

    for (uint k0 = 0; k0 < K; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < BM * BK / 8; i += nthreads) {           // x tile, 8 bf16 per thread-op
            uint r = i / (BK / 8), c = (i % (BK / 8)) * 8;
            uint4 v = uint4(0);
            if (m0 + r < M) v = *(const device uint4*)(x + (m0 + r) * K + k0 + c);
            *(threadgroup uint4*)(&Xs[r][c]) = v;
        }
        for (uint i = tid; i < BN * BK / 8; i += nthreads) {           // weight tile: one uint32 = 8 weights
            uint r = i / (BK / 8), c = (i % (BK / 8)) * 8, n = n0 + r;
            float s = 0.0f, b = 0.0f;
            uint32_t wq = 0;
            if (n < N) {
                wq = w[n * (K / 8) + (k0 + c) / 8];
                s = float(scales[n * (K / 64) + k0 / 64]);
                b = float(biases[n * (K / 64) + k0 / 64]);
            }
            for (int j = 0; j < 8; j++) Ws[r][c + j] = bfloat(s * float((wq >> (4 * j)) & 0xF) + b);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short kk = 0; kk < BK / 16; kk++) {
            for (short i = 0; i < 2; i++)
                for (short j = 0; j < 4; j++) ca[i * 4 + j] = Xs[wm * 16 + fm + 8 * i][kk * 16 + fn + j];
            for (short n = 0; n < 2; n++)
                for (short i = 0; i < 2; i++)
                    for (short j = 0; j < 4; j++)
                        cb[n * 8 + i * 4 + j] = Ws[wn * 32 + n * 16 + fm + 8 * i][kk * 16 + fn + j];
            op.run(ca, cb, cc);
        }
    }
    for (short n = 0; n < 2; n++)
        for (short i = 0; i < 2; i++) {
            uint m = m0 + wm * 16 + fm + 8 * i;
            if (m >= M) continue;
            for (short j = 0; j < 4; j++) {
                uint col = n0 + wn * 32 + n * 16 + fn + j;
                if (col < N) out[m * N + col] = bfloat(cc[n * 8 + i * 4 + j]);
            }
        }
"""
_qmm_k = mx.fast.metal_kernel(name="qmm4_nax", header=_HDR_NAX, source=_QMM_SRC,
                              input_names=["x", "w", "scales", "biases"], output_names=["out"])


def qmm(x, packed, scales, biases, wm=None, wn=4):
    M, Kd = x.shape
    N = packed.shape[0]
    wm = wm or (1 if M <= 16 else 2 if M <= 32 else 4)
    BM, BN = 16 * wm, 32 * wn
    return _qmm_k(inputs=[x, packed, scales, biases], template=[("K", Kd), ("N", N), ("WM", wm), ("WN", wn)],
                  grid=(-(-N // BN) * 32 * wm * wn, -(-M // BM), 1), threadgroup=(32 * wm * wn, 1, 1),
                  output_shapes=[(M, N)], output_dtypes=[x.dtype])[0]


def qmv(x, packed, scales, biases, rows=4, simds=2):
    M, Kd = x.shape
    N = packed.shape[0]
    tgs = -(-N // (rows * simds))
    return _qmv_k(inputs=[x, packed, scales, biases], template=[("M", M), ("K", Kd), ("N", N), ("ROWS", rows),
                                                                ("SIMDS", simds)],
                  grid=(tgs * simds * 32, 1, 1), threadgroup=(simds * 32, 1, 1),
                  output_shapes=[(M, N)], output_dtypes=[x.dtype])[0]


def qlinear(x, packed, scales, biases):
    """x [M, K] bf16 @ W^T for 4-bit W [N, K]: fused unpack-and-dot GEMV for tiny M, tensor-op GEMM
    (unpack into threadgroup memory) for decode-batch M, dequantize + dense matmul for prefill-sized M."""
    M = x.shape[0]
    if M <= QMV_MAX_M:
        return qmv(x, packed, scales, biases)
    if M <= QMM_MAX_M and has_nax():
        return qmm(x, packed, scales, biases)
    # ponytail: prefill-sized M is compute-bound and MLX's GEMM beats qmm (which re-unpacks W per M-tile);
    # unpacking once costs ~1% here. A qmm with larger tiles / W reuse across M-tiles is the upgrade path.
    return x @ dequantize4(packed, scales, biases).T


# ---------------------------------------------------------------- bf16 GEMM for wide matrices at decode batch sizes
# MLX's GEMM streams weights at ~135 GB/s up to 12 rows but only ~95-110 GB/s at 13-16 rows. On *wide* weights
# (gate|up: 12288 rows, lm_head: 151936) the qmm tiling with a plain bf16 weight tile has enough threadgroups to
# beat it; on narrow ones (o_proj: 16 threadgroups) it loses, so callers gate it by N.
_BMM_SRC = _QMM_SRC[:_QMM_SRC.index("        for (uint i = tid; i < BN * BK / 8; i += nthreads) {           // weight tile")] + """
        for (uint i = tid; i < BN * BK / 8; i += nthreads) {
            uint r = i / (BK / 8), c = (i % (BK / 8)) * 8, n = n0 + r;
            uint4 v = uint4(0);
            if (n < N) v = *(const device uint4*)(w + n * K + k0 + c);
            *(threadgroup uint4*)(&Ws[r][c]) = v;
        }
""" + _QMM_SRC[_QMM_SRC.index("        threadgroup_barrier(mem_flags::mem_threadgroup);\n        for (short kk = 0;"):]
_bmm_k = mx.fast.metal_kernel(name="bmm_nax", header=_HDR_NAX, source=_BMM_SRC, input_names=["x", "w"],
                              output_names=["out"])


def bmm(x, w, wn=4):
    """x [M <= 16, K] @ w[N, K]^T in bf16."""
    M, Kd = x.shape
    N = w.shape[0]
    BN = 32 * wn
    return _bmm_k(inputs=[x, w], template=[("K", Kd), ("N", N), ("WM", 1), ("WN", wn)],
                  grid=(-(-N // BN) * 32 * wn, 1, 1), threadgroup=(32 * wn, 1, 1),
                  output_shapes=[(M, N)], output_dtypes=[x.dtype])[0]


def use_bmm(M, N):
    """End-to-end A/B on Qwen3-1.7B decode steps (min of 3 interleaved trials): at 13-16 rows, gate|up and
    lm_head through bmm cut the step 41.7 -> 38.6 ms; down/o/qkv through bmm are neutral-to-worse; at <= 12
    rows MLX's GEMM is already at full bandwidth."""
    return 13 <= M <= 16 and N >= 8192 and has_nax()
