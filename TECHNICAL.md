# GLM-5.2 on RTX 4090 — technical write-up

How the DeepSeek-Sparse-Attention (DSA) kernel stack was ported from Hopper/Blackwell to Ada (sm_89), and how the last MoE bug was found and fixed. See [README.md](README.md) for the high-level result and usage.

## Why GLM-5.2 doesn't run on Ada out of the box

GLM-5.2 (753B, MoE) uses **DeepSeek Sparse Attention**: a lightweight "lightning indexer" (an FP8 GEMM producing per-(query, key) logits) selects the top-k=2048 keys, then a sparse MLA attention attends only to them. Stock implementations route this through a stack of kernels with no Ada code path:

| Stage | Stock kernel | Arch gate |
|---|---|---|
| Indexer logits | DeepGEMM `fp8_mqa_logits` / `fp8_paged_mqa_logits` | SM90+ (WGMMA/TMA) |
| Top-k + page transforms | `sgl_kernel.fast_topk*` | SM100+ (tcgen05/TMEM) |
| MLA sparse decode | FlashMLA / tilelang v2 | SM90+ (WGMMA) |

Bypass one and the next fires — there is no portable fallback (cf. upstream sglang #23657, ktransformers #1885, vLLM #30644 / #45317 / #35021). GLM-5.2-FP8's weights are *standard* `e4m3` block-fp8 (`weight_block_size=[128,128]`) — the ordinary Ada-OK W8A8 path — so the weight format is **not** the wall; the DSA kernel stack is.

## What the port does

`ada_dsa.py` (`apply_patches()`) monkeypatches the SM90+/SM100 symbols with portable equivalents, only when `device_capability < (9, 0)`:

1. **Indexer logits** → a Triton kernel that bf16-upcasts the fp8 operands so `tl.dot` runs on any TensorCore arch (the fp8 values are exact in bf16). Ported from ROCm/aiter's arch-neutral kernel; the paged variant adds an eager per-batch gather over the paged cache.
2. **Top-k** → `torch.topk` over a per-row windowed view.
3. **Fused page transforms** (logical → physical, `page_size=64`) → portable torch: a `page_table[req, logical]` gather (paged / decode) and `logical + offset` (ragged / extend). Needed because the topk indices are in logical-token space but the kernel gathers from the physical paged KV pool.
4. **MLA sparse decode** → the build's default tilelang kernel needs Hopper WGMMA (`wg_wait`); route to the **non-WGMMA `_v1` kernel**, dequantize the packed fp8-656 cache to bf16-576 with sglang's own `dequantize_k_cache`, and use `block_I=32, threads=128` so it fits Ada's ~99 KB dynamic-shared-memory cap (the default `block_I=64` needs ~108 KB).

The packed KV layout, for reference: each 656-byte token is `[512 nope-fp8] + [16 B = 4×fp32 per-128 tile scales] + [128 B = 64×bf16 rope]`; dequant gives `[512 nope-bf16] + [64 rope-bf16]` = 576.

## Verification

Every ported kernel is validated against a reference on sm_89 (scripts in [`verify/`](verify/)):

| What | Test | Result |
|---|---|---|
| Indexer `fp8_mqa_logits` | vs DeepGEMM torch reference | **~1e-7, 100% top-k agreement** |
| MLA decode (v1 + `block_I=32`) | vs manual MLA-absorb reference | **cosine 0.999997** |
| `dequant_k_cache` (656→576) | fast-path vs reference | **cosine 1.0** |
| KV-cache write (quant→dequant round-trip) | vs input | **cosine 0.9997** |
| Dense block-fp8 GEMM | vs bf16 | **cosine 0.9993** |
| **Full attention on the live model's real tensors** | in-shim self-check vs torch MLA ref | **cosine 0.999999** |

## The MoE fix (`--disable-shared-experts-fusion`)

With the DSA stack ported, attention is correct (0.999999 on live tensors) but output was still incoherent. A per-layer residual-norm trace showed a *structurally healthy* forward — finite, smooth growth, normal "massive-activation" spikes, continuous across pipeline boundaries — i.e. a subtle value error, not an explosion or a transfer corruption.

Root cause: GLM-5.2 has `n_shared_experts=1`, and sglang by default **fuses the shared expert into the routed-expert grouped fp8 MoE GEMM**, whose fused path is wrong on sm_89. (The standalone *dense* block-fp8 GEMM is correct — it's specifically the grouped/fused MoE kernel with the shared expert folded in.)

Fix: **`--disable-shared-experts-fusion`**. The shared expert then runs as a separate (verified-correct) dense GEMM → fully coherent output. The routed experts go through the grouped GEMM and are correct too, so the bug was specifically the *fusion*.

## Notes on the test deployment

- 24× RTX 4090-48GB, 3 nodes × 8, TP=8 × PP=3, over a 10 GbE inter-node link.
- `--disable-cuda-graph` is required: the stock cuda-graph decode-metadata kernel is SM90+, and the portable paged-logits path runs eager.
- NCCL transport (`NCCL_P2P_DISABLE` / `NCCL_IB_DISABLE`) should be set to match your interconnect.
- Validated on GLM-5.2-FP8: 78 layers, MLA head 576 (= 512 nope + 64 rope), `page_size=64`, 256 routed + 1 shared expert.

## Upstreaming

Two clean contributions: (1) the portable DSA kernel stack here — closes sglang #23657 / ktransformers #1885 for sub-Hopper NVIDIA (and applies to `sm_120` consumer Blackwell, which the stock `sm_90`/`sm_100` kernels also miss); (2) a `sm < 90` guard that auto-disables shared-expert fusion, or a fix for the fused grouped-GEMM on Ada.
