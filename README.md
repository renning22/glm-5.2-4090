# GLM-5.2 on RTX 4090

**Running the full [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2-FP8) — the 753B-parameter SOTA open-weights model — in native FP8 on consumer NVIDIA RTX 4090 GPUs.** As far as we know, the first time a DeepSeek-Sparse-Attention (DSA) model has run correctly on Ada (sm_89). The stock sglang / vLLM stack hard-requires Hopper (H100/H200) or Blackwell (B200) for GLM-5.2 — its sparse-attention kernels are gated to `sm_90` / `sm_100` with no Ada fallback. This repo is a drop-in `ada_dsa.py` that ports that whole kernel stack to the 4090.

```
prompt:  "The capital of France is"
output:  "Paris. Distance from Paris to Lyon is 391 km, while direct flight time is 1 h 5 min"
```

## Highlights

- **Full 753B model, full FP8** — not a distilled or int4 variant. The complete GLM-5.2-FP8 weights.
- **24× RTX 4090-48GB** (3 nodes × 8), pipeline + tensor parallel — proven, coherent chat / reasoning / code.
- **~10 tokens/sec single-stream** (CUDA-graph) — interactive speed for the full 753B on commodity cards.
- **Every ported kernel validated** against a reference, down to ~1e-6 — including **0.999999** cosine on the live model's real tensors.
- **Open** — the kernels, the one-call installer, and the verification scripts are all here.

## Hardware sizing

The FP8 weights are ~**753 GB**, so the model has to be split across enough cards to hold the weights plus KV-cache and activations. Roughly:

| GPU | VRAM/card | GPUs needed | Layout | Status |
|---|---|---|---|---|
| RTX 4090 | 48 GB | **24** | TP=8 × PP=3 (3 nodes) | ✅ proven (this repo) |
| RTX 4090 | 24 GB | ~40–48 | TP=8 × PP=5–6 (5–6 nodes) | sizing estimate |
| RTX 5090 | 32 GB | ~32 | TP=8 × PP=4 (4 nodes) | sizing estimate¹ |

¹ The RTX 5090 is `sm_120` (consumer Blackwell), which the stock `sm_90`/`sm_100` DSA kernels also don't cover — so it needs this same port (widen the capability guard to include `sm_120`). Only the 4090-48GB config is tested here; the others are VRAM-fit estimates (assume ~6–8 GB/card reserved for KV + activations + CUDA context, more for larger context windows).

## Usage

Tested against an sglang build with the `nsa` / tilelang DSA backend, in an environment where `tilelang` is available (we grafted tilelang 0.1.11 + tvm-ffi from [KTransformers](https://github.com/kvcache-ai/ktransformers)).

1. Put `ada_dsa.py` on `PYTHONPATH`.
2. Append the guard to `sglang/srt/layers/attention/nsa/nsa_indexer.py` (after its `import deep_gemm`):

   ```python
   import torch
   if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 9:
       import ada_dsa; ada_dsa.apply_patches()
   ```

3. Launch (example: 24× RTX 4090-48GB, TP=8 × PP=3):

   ```bash
   export SGLANG_NSA_FUSE_TOPK=1            # use the ported fused page-mapping transforms
   export SGLANG_ENABLE_JIT_DEEPGEMM=0

   python -m sglang.launch_server \
     --model-path zai-org/GLM-5.2-FP8 \
     --tp-size 8 --pp-size 3 --nnodes 3 --dist-init-addr <rank0-ip>:30200 \
     --trust-remote-code --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85 \
     --attention-backend nsa --nsa-decode-backend tilelang --nsa-prefill-backend tilelang \
     --fp8-gemm-backend triton \
     --disable-shared-experts-fusion \      # REQUIRED on Ada (the MoE fix), see TECHNICAL.md
     --tool-call-parser glm47 --reasoning-parser glm45 \
     --node-rank <0|1|2> --host 0.0.0.0 --port 8000
   ```

`--disable-shared-experts-fusion` is required on Ada. CUDA-graph is on by default here and gives the full speed (about 10 tok/s single-stream vs about 2.5 in eager); it needs one extra one-line guard in `deep_gemm_wrapper/entrypoint.py` (see TECHNICAL.md). If you'd rather not patch that, add `--disable-cuda-graph` and run eager. Configure NCCL transport (`NCCL_P2P_DISABLE` / `NCCL_IB_DISABLE`) to match your fabric.

## How it works

`ada_dsa.py` monkeypatches GLM-5.2's SM90+/SM100-only DSA kernels — the lightning-indexer GEMM, the top-k + page-mapping, and the MLA sparse decode — with portable Triton + a non-WGMMA tilelang path, only on sub-Hopper GPUs. Plus one config fix for the MoE. **Full write-up, kernel-by-kernel walkthrough, and the verification table: [TECHNICAL.md](TECHNICAL.md).**

## Status

GLM-5.2 runs *correctly* on consumer hardware where the stock stack hard-crashes, at **about 10 tokens/sec single-stream** (CUDA-graph; about 2.5 in eager mode). That's interactive speed for the full 753B model on commodity cards. The portable indexer / top-k / page-transform stack is model-agnostic and should apply to other DSA models (e.g. DeepSeek-V3.2-style) with minor adjustment.

## License

Apache-2.0.

---

*Built by [@renning22](https://github.com/renning22).*
