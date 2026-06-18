import torch, torch.nn.functional as F
from sglang.srt.layers.quantization.fp8_kernel import w8a8_block_fp8_matmul
from sglang.srt.layers.quantization.fp8_utils import per_block_cast_to_fp8
try:
    from sglang.srt.layers.quantization.fp8_kernel import per_token_group_quant_fp8
except Exception:
    from sglang.srt.layers.quantization.fp8_kernel import per_token_group_quant_8bit as _p
    def per_token_group_quant_fp8(x, g): return _p(x, g, dst_dtype=torch.float8_e4m3fn)
torch.manual_seed(0); dev="cuda"; cos=0.0
for (M,K,N) in [(256,512,384),(128,2048,1024)]:
    A = torch.randn(M, K, device=dev, dtype=torch.bfloat16)*0.3
    W = torch.randn(N, K, device=dev, dtype=torch.bfloat16)*0.3
    ref = (A.float() @ W.float().t())
    A_fp8, As = per_token_group_quant_fp8(A, 128)
    W_fp8, Ws = per_block_cast_to_fp8(W)
    got = w8a8_block_fp8_matmul(A_fp8, W_fp8, As, Ws, [128,128], output_dtype=torch.bfloat16)
    g = got.float().flatten(); r = ref.flatten()
    cos = F.cosine_similarity(g, r, dim=0).item()
    rel = ((g-r).abs()/r.abs().clamp(0.1)).mean().item()
    print("M%d K%d N%d: cosine=%.6f mean_rel=%.3e finite=%s" % (M,K,N,cos,rel,bool(torch.isfinite(got).all())))
print("VERDICT:", "FP8 GEMM OK" if cos>0.99 else "FP8 BLOCK GEMM BROKEN ON ADA (this venv)")
