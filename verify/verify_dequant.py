import torch
from sglang.srt.layers.attention.nsa.dequant_k_cache import _dequantize_k_cache_fast_wrapped, _dequantize_k_cache_ref
torch.manual_seed(1)
N = 64
# build a packed-656 cache: [512 nope-fp8] + [16B=4 fp32 scales] + [128B=64 bf16 rope]
nope = (torch.randn(N, 512, device="cuda") * 0.3).to(torch.float8_e4m3fn)
scales = (torch.rand(N, 4, device="cuda") * 0.5 + 0.1).to(torch.float32)
rope = (torch.randn(N, 64, device="cuda") * 0.3).to(torch.bfloat16)
packed = torch.empty(N, 656, dtype=torch.float8_e4m3fn, device="cuda")
packed[:, :512] = nope
packed[:, 512:528] = scales.view(torch.float8_e4m3fn).view(N, 16)
packed[:, 528:656] = rope.view(torch.float8_e4m3fn).view(N, 128)
cache = packed.view(N, 1, 656)
fast = _dequantize_k_cache_fast_wrapped(cache)   # (N,1,576) bf16
ref  = _dequantize_k_cache_ref(cache)            # (N,1,576) bf16
f = fast.float().flatten(); r = ref.float().flatten()
print("fast vs ref: max_abs=%.3e cosine=%.6f finite=%s" % ((f-r).abs().max().item(),
      torch.nn.functional.cosine_similarity(f, r, dim=0).item(), bool(torch.isfinite(fast).all())))
# spot check: dequant nope should be ~ nope_fp8 * scale_per_128_tile
print("VERDICT:", "DEQUANT OK" if torch.nn.functional.cosine_similarity(f,r,dim=0).item()>0.999 else "DEQUANT MISMATCH")
