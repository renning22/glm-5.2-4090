import torch
from sglang.srt.layers.attention.nsa.quant_k_cache import quantize_k_cache
from sglang.srt.layers.attention.nsa.dequant_k_cache import dequantize_k_cache
torch.manual_seed(0); dev="cuda"
N = 256
kv = torch.randn(N, 1, 1, 576, device=dev, dtype=torch.bfloat16)   # (blocks, block_size, 1, 576)
q656 = quantize_k_cache(kv)              # -> (N,1,1,656) fp8 packed
deq  = dequantize_k_cache(q656)          # -> (N,1,1,576) bf16
a = kv.float().flatten(); b = deq.float().flatten()
rel = ((a-b).abs() / a.abs().clamp(0.05)).mean().item()
cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
kvn, deqn = kv[...,:512].float(), deq[...,:512].float()
kvr, deqr = kv[...,512:].float(), deq[...,512:].float()
print("q656 shape=%s dtype=%s" % (tuple(q656.shape), q656.dtype))
print("roundtrip: cosine=%.6f mean_rel=%.3e | nope cos=%.6f | rope cos=%.6f" % (
    cos, rel,
    torch.nn.functional.cosine_similarity(kvn.flatten(), deqn.flatten(), dim=0).item(),
    torch.nn.functional.cosine_similarity(kvr.flatten(), deqr.flatten(), dim=0).item()))
print("VERDICT:", "KV WRITE OK" if cos > 0.99 else "KV WRITE/QUANT BROKEN ON ADA")
