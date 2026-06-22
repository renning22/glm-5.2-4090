# Tuned block-FP8 GEMM configs for RTX 4090 (sm_89)

sglang's W8A8 block-FP8 triton kernel reads a per-`(N, K, device, block_shape)` tuned config. It ships
them for H100 / H200 / B200 / A100 — but **none for the RTX 4090**, so on Ada it falls back to a generic
default and logs `Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal!`. On the
4090 that default is genuinely bad.

These are tuned configs for the 9 dense GEMM shapes GLM-5.2 uses per layer (MLA q/kv projections, fused
QKV `N=2624,K=6144`, dense-MLP gate/up/down, output proj, lightning indexer `N=128/512`).

**Measured effect: ~+17% decode aggregate throughput, and ~+90% at 16K context** (where the dense GEMMs
dominate). Lossless — pure kernel scheduling, no change to weights or numerics.

## Install

```bash
cp *.json <sglang>/srt/layers/quantization/configs/
```

sglang auto-loads them at startup, matched by `N`, `K`, `device_name`, `dtype`, `block_shape`. Restart
the server; the "sub-optimal" warning should disappear for these shapes.

## Regenerating / tuning other shapes

Run sglang's `benchmark/kernels/quantization/tuning_block_wise_kernel.py --N <n> --K <k> --input-type fp8`
once per shape. To find which shapes your model leaves untuned, grep the server startup log for:

```
Config file not found at ...device_name=NVIDIA_GeForce_RTX_4090...
```

Pruning the tuner's `get_configs_compute_bound()` grid to the decode regime (`block_m ∈ {16,32,64,128}`,
`block_k = 128`, fewer stages/groups) cuts tuning time ~9× to a few minutes per shape with no loss on
these shapes.
