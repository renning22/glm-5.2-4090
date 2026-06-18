# GLM-5.2 on RTX 4090 — porting the DSA kernel stack to Ada (sm_89)

**Running [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2-FP8) — a DeepSeek-Sparse-Attention (DSA) model — on consumer NVIDIA RTX 4090 GPUs. First known correct serving of a DSA model on Ada (sm_89).**

The stock sglang / vLLM stack hard-requires **Hopper (H100/H200) or Blackwell (B200)** for GLM-5.2: its sparse-attention path is a stack of kernels gated to `sm_90` (WGMMA/TMA) and `sm_100` (tcgen05/TMEM), with no Ada code path. This repo is a **drop-in `ada_dsa.py`** that ports that whole stack to Ada with portable Triton + a non-WGMMA tilelang kernel — plus the one extra config flag that makes the MoE correct on Ada.

Result on 3× RTX 4090-48G nodes (PP=3 × TP=8): **coherent chat, reasoning, and code generation.**

```
prompt:  "The capital of France is"
output:  "Paris. Distance from Paris to Lyon is 391 km, while direct flight time is 1 h 5 min"

prompt:  "def fibonacci(n):"
output:  "if n == 0: return 0
          elif n == 1: return 1
          else: return fibonacci(n-1) + fibonacci(n-2)"
```

---

## Why GLM-5.2 doesn't run on Ada out of the box

GLM-5.2 uses **DeepSeek Sparse Attention**: a lightweight "lightning indexer" (an FP8 GEMM producing per-(query, key) logits) selects the top-k=2048 keys, then a sparse MLA attention attends only to them. Stock implementations route this through:

| Stage | Stock kernel | Arch gate |
|---|---|---|
| Indexer logits | DeepGEMM `fp8_mqa_logits` / `fp8_paged_mqa_logits` | SM90+ (WGMMA/TMA) |
| Top-k + page transforms | `sgl_kernel.fast_topk*` | SM100+ (tcgen05/TMEM) |
| MLA sparse decode | FlashMLA / tilelang v2 | SM90+ (WGMMA) |

Bypass one and the next fires — there is no portable fallback. (See upstream sglang #23657, ktransformers #1885, vLLM #30644 / #45317 / #35021.) GLM-5.2-FP8's weights are *standard* `e4m3` block-fp8 (`weight_block_size=[128,128]`) — the ordinary Ada-OK W8A8 path — so the weight format is **not** the wall; the DSA kernel stack is.

## What this port does

`ada_dsa.py` monkeypatches the SM90+/SM100 symbols with portable equivalents, only when `device_capability < (9, 0)`:

1. **Indexer logits** → a Triton kernel that bf16-upcasts the fp8 operands so `tl.dot` runs on any TensorCore arch (the fp8 values are exact in bf16). Ported from ROCm/aiter's arch-neutral kernel.
2. **Top-k** → `torch.topk` over a per-row windowed view.
3. **Fused page transforms** (logical→physical, `page_size=64`) → portable torch: `page_table[req, logical]` gather (paged/decode) and `logical + offset` (ragged/extend).
4. **MLA sparse decode** → the build's default tilelang kernel needs Hopper WGMMA (`wg_wait`); route to the **non-WGMMA `_v1` kernel**, dequantize the packed fp8-656 cache to bf16-576 with sglang's own `dequantize_k_cache`, and use `block_I=32, threads=128` so it fits Ada's ~99 KB dynamic-shared-memory cap.

Plus one stock-sglang fix (see below): **`--disable-shared-experts-fusion`**.

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

With the DSA stack ported, attention is correct (0.999999 on live tensors) but output was still incoherent. A per-layer residual-norm trace showed a *structurally healthy* forward (finite, continuous across pipeline boundaries) — i.e. a subtle value error, not an explosion. Root cause: GLM-5.2 has `n_shared_experts=1`, and sglang by default **fuses the shared expert into the routed-expert grouped fp8 MoE GEMM**, whose fused path is wrong on sm_89 (the standalone *dense* block-fp8 GEMM is correct). Passing **`--disable-shared-experts-fusion`** runs the shared expert as a separate (verified-correct) dense GEMM → fully coherent output.

## Usage

Tested on a build of sglang with the `nsa` / tilelang DSA backend, on an environment with `tilelang` available (we grafted tilelang 0.1.11 + tvm-ffi from [KTransformers](https://github.com/kvcache-ai/ktransformers)).

1. Put `ada_dsa.py` on `PYTHONPATH`.
2. Append the guard to `sglang/srt/layers/attention/nsa/nsa_indexer.py` (after its `import deep_gemm`):

   ```python
   import torch
   if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 9:
       import ada_dsa; ada_dsa.apply_patches()
   ```

3. Launch (3 nodes, PP=3 × TP=8 for the 744 GB model; adjust to your fleet):

   ```bash
   # env that matters
   export SGLANG_NSA_FUSE_TOPK=1            # use the ported fused page-mapping transforms
   export SGLANG_ENABLE_JIT_DEEPGEMM=0
   export NCCL_P2P_DISABLE=1                # required on RTX 4090-48G (P2P corrupts)
   export NCCL_IB_DISABLE=1

   python -m sglang.launch_server \
     --model-path zai-org/GLM-5.2-FP8 \
     --tp-size 8 --pp-size 3 --nnodes 3 --dist-init-addr <rank0-ip>:30200 \
     --trust-remote-code --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85 \
     --attention-backend nsa --nsa-decode-backend tilelang --nsa-prefill-backend tilelang \
     --fp8-gemm-backend triton --disable-cuda-graph \
     --disable-shared-experts-fusion \      # REQUIRED on Ada (see "The MoE fix")
     --tool-call-parser glm47 --reasoning-parser glm45 \
     --node-rank <0|1|2> --host 0.0.0.0 --port 8000
   ```

`--disable-cuda-graph` is required (the stock cuda-graph decode-metadata kernel is SM90+, and the portable paged-logits path runs eager).

## Status & caveats

- **Capability, not yet throughput.** Single-stream decode is ~**2.5 tok/s** in this config (eager mode, pipeline-parallel over TCP). This is a "it runs *correctly*, where the stock stack hard-crashes" result; cuda-graph capture and a faster interconnect are future work.
- Validated on GLM-5.2-FP8 (78 layers, MLA head 576 = 512 nope + 64 rope, `page_size=64`, 256 routed + 1 shared expert).
- The portable indexer/top-k/page-transform stack is model-agnostic — it should apply to other DSA models (e.g. DeepSeek-V3.2-style) with minor adjustment.

## Upstreaming

Two clean contributions: (1) the portable DSA kernel stack here (closes sglang #23657 / ktransformers #1885 for sub-Hopper NVIDIA); (2) a `sm < 90` guard that auto-disables shared-expert fusion (or fixes the fused grouped-GEMM on Ada).

## License

Apache-2.0.

---

*Built by [@renning22](https://github.com/renning22). The DSA-on-Ada port and verification were developed against a live 3×RTX-4090 deployment.*
