"""Verify the tilelang v1 sparse MLA decode kernel (block_I=32, threads=128) against a
manual MLA-absorb reference on sm_89. If this matches, the decode is correct and the
incoherence is elsewhere (prefill); if not, v1 is the numerical bug."""
import torch
from sglang.srt.layers.attention.nsa.tilelang_kernel import sparse_attention_fwd_kernel_v1

torch.manual_seed(0)
dev = "cuda"
N, H, D, TAIL = 4, 8, 512, 64        # 4 query tokens, 8 heads, 512 nope + 64 rope
Skv = 256
topk = 2048                          # kernel requires topk==2048
sm_scale = 1.0 / (D + TAIL) ** 0.5

q  = torch.randn(N, H, D + TAIL, device=dev, dtype=torch.bfloat16) * 0.5
kv = torch.randn(Skv, 1, D + TAIL, device=dev, dtype=torch.bfloat16) * 0.5
# indices: each query attends to a causal-ish window; -1 = masked slot
idx = torch.full((N, 1, topk), -1, device=dev, dtype=torch.int32)
valid_len = []
for i in range(N):
    L = Skv - (N - 1 - i) * 8        # token i sees first L kv tokens
    valid_len.append(L)
    idx[i, 0, :L] = torch.arange(L, device=dev, dtype=torch.int32)

kern = sparse_attention_fwd_kernel_v1(H, D, TAIL, topk, sm_scale=sm_scale, num_stages=1, block_I=32, threads=128)
out = kern(q.unsqueeze(0), kv.unsqueeze(0), idx.unsqueeze(0))[0]   # [N,H,D]

# reference MLA: score = q·kv over full 576, softmax over valid window, out = attn·kv[:,:512]
qf = q.float(); kvf = kv[:, 0, :].float()          # [Skv,576]
ref = torch.empty(N, H, D, device=dev, dtype=torch.float32)
for i in range(N):
    L = valid_len[i]
    s = torch.einsum("hd,nd->hn", qf[i], kvf[:L]) * sm_scale       # [H,L]
    a = torch.softmax(s, dim=-1)
    ref[i] = a @ kvf[:L, :D]                                       # [H,512]

og = out.float()
rel = ((og - ref).abs() / ref.abs().clamp(1e-2)).mean().item()
mx  = (og - ref).abs().max().item()
cos = torch.nn.functional.cosine_similarity(og.flatten(), ref.flatten(), dim=0).item()
print(f"valid_lens={valid_len}")
print(f"out finite={bool(torch.isfinite(out).all())} mean_rel={rel:.3e} max_abs={mx:.3e} cosine={cos:.6f}")
print("VERDICT:", "DECODE KERNEL CORRECT ✓" if cos > 0.99 else "DECODE KERNEL WRONG ✗ (this is the bug)")
