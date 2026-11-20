"""CUDA backend checks (skipped without an NVIDIA GPU). Run first on the rented box:
    uv run --with torch --with triton pytest tests/test_cuda.py -q
Reference outputs come from torch SDPA and from HF transformers' Qwen3 (reference only)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
BS, D = 16, 128


def make(ctx_lens, HKV, H):
    rng = np.random.default_rng(0)
    nb = sum(-(-c // BS) for c in ctx_lens) + 4
    perm = rng.permutation(nb)
    bt, i = np.zeros((len(ctx_lens), max(-(-c // BS) for c in ctx_lens)), np.int32), 0
    for s, c in enumerate(ctx_lens):
        n = -(-c // BS)
        bt[s, :n] = perm[i:i + n]; i += n
    kc = torch.randn(nb * BS, HKV, D, device="cuda", dtype=torch.bfloat16)
    vc = torch.randn_like(kc)
    q = torch.randn(len(ctx_lens), H, D, device="cuda", dtype=torch.bfloat16)
    return q, kc, vc, torch.from_numpy(bt).cuda(), torch.tensor(ctx_lens, dtype=torch.int32, device="cuda")


@pytest.mark.parametrize("impl", ["triton", "cuda"])
@pytest.mark.parametrize("H,HKV", [(16, 8), (32, 8)])
def test_paged_decode(impl, H, HKV):
    from engine.backend_cuda import paged_decode_reference
    mod = __import__(f"engine.kernels_{impl}", fromlist=["paged_decode"])
    ctx = [1, BS - 1, BS, BS + 1, 300, 4097]
    q, kc, vc, bt, cl = make(ctx, HKV, H)
    out = mod.paged_decode(q, kc, vc, bt, cl, BS)
    ref = paged_decode_reference(q, kc, vc, bt, cl, BS)
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("attn", ["torch", "triton", "cuda"])
def test_model_matches_transformers(attn):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from engine.backend_cuda import TorchBackend
    from engine.model import Qwen3
    from tests.test_model import run_seq
    M = "Qwen/Qwen3-0.6B"
    tok = AutoTokenizer.from_pretrained(M)
    ids = tok.encode("The quick brown fox jumps over the lazy dog. " * 8)
    ref = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16).cuda()
    m = Qwen3(M, TorchBackend(attn=attn))
    cache = m.B.alloc_kv(m.n_layers, 64 * BS, m.Hkv, m.D)
    for s in range(0, len(ids), 7):                    # chunked prefill, then the decode path
        ours = run_seq(m, cache, ids[s:s + 7], s, list(range(64))[::-1])
    with torch.no_grad():
        theirs = ref(torch.tensor([ids], device="cuda")).logits[0, -1].float()
    cos = torch.nn.functional.cosine_similarity(ours[None].float(), theirs[None], dim=-1).item()
    assert cos > 0.999 and ours.argmax().item() == theirs.argmax().item(), cos
