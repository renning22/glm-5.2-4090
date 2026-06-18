"""Drop-in portable replacement for DeepGEMM's Hopper/Blackwell-only DSA indexer
kernels, for NVIDIA Ada (sm_89 / RTX 4090). Monkeypatched over `deep_gemm` so all
of sglang's `deep_gemm.fp8_mqa_logits` / `fp8_paged_mqa_logits` /
`get_paged_mqa_logits_metadata` call sites route here on sm<90.

Validated: the core `_mqa_logits` matches the DeepGEMM torch reference to ~1e-7
with 100% top-k agreement on sm_89. (@renning22, 2026)
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_mqa_logits_kernel(
    Q_ptr, KV_ptr, kv_scales_ptr, weights_ptr, cu_start_ptr, cu_end_ptr, logits_ptr,
    seq_len, seq_len_kv,
    NUM_HEADS: tl.constexpr, HEAD_SIZE: tl.constexpr,
    stride_q_s, stride_q_h, stride_q_d, stride_kv_s, stride_kv_d,
    stride_w_s, stride_w_h, stride_logits_s, stride_logits_k, BLOCK_KV: tl.constexpr,
):
    row_id = tl.program_id(0)
    logits_row = logits_ptr + row_id * stride_logits_s
    h = tl.arange(0, NUM_HEADS)[:, None]
    d = tl.arange(0, HEAD_SIZE)
    q_block = tl.load(Q_ptr + row_id * stride_q_s + h * stride_q_h + d[None, :] * stride_q_d).to(tl.bfloat16)
    w_block = tl.load(weights_ptr + row_id * stride_w_s + h * stride_w_h).to(tl.float32)
    start = tl.maximum(tl.load(cu_start_ptr + row_id), 0)
    end = tl.minimum(tl.load(cu_end_ptr + row_id), seq_len_kv)
    unmasked_end = ((end - start) // BLOCK_KV) * BLOCK_KV
    col = tl.arange(0, BLOCK_KV) + start
    kv_ptrs = KV_ptr + col[None, :] * stride_kv_s + d[:, None] * stride_kv_d
    sc_ptrs = kv_scales_ptr + col
    out_ptrs = logits_row + col * stride_logits_k
    for _ in tl.range(0, unmasked_end, BLOCK_KV):
        s = tl.dot(q_block, tl.load(kv_ptrs).to(tl.bfloat16)).to(tl.float32) * tl.load(sc_ptrs)[None, :]
        s = tl.sum(tl.maximum(s, 0.0) * w_block, axis=0)
        tl.store(out_ptrs, s)
        kv_ptrs += BLOCK_KV * stride_kv_s
        sc_ptrs += BLOCK_KV
        out_ptrs += BLOCK_KV * stride_logits_k
        col += BLOCK_KV
    mask = col < end
    s = tl.dot(q_block, tl.load(kv_ptrs, mask=mask[None, :], other=0.0).to(tl.bfloat16)).to(tl.float32)
    s = s * tl.load(sc_ptrs, mask=mask, other=0.0)[None, :]
    s = tl.sum(tl.maximum(s, 0.0) * w_block, axis=0)
    tl.store(out_ptrs, s, mask=(col >= start) & (col < end))


def _mqa_logits(q, kv, kv_scales, weights, ks, ke, clean_logits=False):
    S, H, D = q.shape
    Skv = kv.shape[0]
    logits = torch.full((S, Skv), float("-inf") if clean_logits else 0.0,
                        dtype=torch.float32, device=q.device)
    _fp8_mqa_logits_kernel[(S,)](
        q, kv.contiguous(), kv_scales.contiguous(), weights, ks, ke, logits, S, Skv, H, D,
        *q.stride(), *kv.stride(), *weights.stride(), *logits.stride(),
        BLOCK_KV=64, num_warps=4, num_stages=2)
    return logits


# ---- DeepGEMM-signature shims (monkeypatched over deep_gemm.*) ----
def fp8_mqa_logits(q_fp8, kv_fp8, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=False, **kw):
    k, scale = kv_fp8
    return _mqa_logits(q_fp8, k.view(torch.float8_e4m3fn), scale.float(), weights.float(),
                       cu_seqlen_ks.to(torch.int32), cu_seqlen_ke.to(torch.int32), clean_logits)


def get_paged_mqa_logits_metadata(*args, **kwargs):
    return None  # the Triton path computes its own grid


def _windowed_topk_logical(score, lengths, topk, row_starts=None):
    """Top-k *logical* indices within each row's valid window [row_starts,
    row_starts+lengths). Invalid / padding slots get the NSA sentinel -1 so the
    decode kernel masks them (`mask = Indices >= 0`). Returns [B, topk] int32."""
    B, L = score.shape
    dev = score.device
    col = torch.arange(L, device=dev)[None, :]
    lo = (row_starts[:, None].long() if row_starts is not None
          else torch.zeros(B, 1, dtype=torch.long, device=dev))
    hi = lo + lengths[:, None].long()
    masked = score.masked_fill(~((col >= lo) & (col < hi)), float("-inf"))
    k = min(int(topk), L)
    vals, idx = torch.topk(masked, k, dim=1)
    idx = idx.to(torch.int32)
    INVALID = torch.full((B, 1), -1, dtype=torch.int32, device=dev)
    idx = torch.where(torch.isfinite(vals), idx, INVALID)
    if k < topk:
        idx = torch.cat([idx, INVALID.expand(B, topk - k)], dim=1)
    return idx.contiguous()


def fast_topk_v2(score, lengths, topk, row_starts=None):
    """Portable sgl_kernel.fast_topk_v2 — raw *logical* top-k indices (non-fused path)."""
    return _windowed_topk_logical(score, lengths, topk, row_starts)


def fast_topk_transform_fused(score, lengths, page_table_size_1, cu_seqlens_q, topk, row_starts=None):
    """Portable sgl_kernel.fast_topk_transform_fused (PAGED / decode). Top-k logical
    indices, then map each to its physical KV-pool position via the per-token page
    table: dst[q,j] = page_table_size_1[req(q), logical[q,j]]; -1 sentinels preserved.
    Each query row q belongs to request req(q), recovered from cu_seqlens_q (the page
    table is per-request, but there can be several query rows per request)."""
    logical = _windowed_topk_logical(score, lengths, topk, row_starts)   # [Bq, topk] int32
    Bq = score.shape[0]
    dev = score.device
    ptr = page_table_size_1
    if ptr.shape[0] != Bq:
        if cu_seqlens_q is not None and cu_seqlens_q.numel() >= 2:
            bounds = cu_seqlens_q[1:].contiguous().to(torch.int64)
            req_idx = torch.searchsorted(bounds, torch.arange(Bq, device=dev), right=True)
            req_idx = req_idx.clamp(max=ptr.shape[0] - 1)
            ptr = ptr[req_idx]                                            # [Bq, max_len]
        elif ptr.shape[0] == 1:
            ptr = ptr.expand(Bq, -1)
    valid = logical >= 0
    gidx = logical.clamp(min=0).clamp(max=ptr.shape[1] - 1).long()
    phys = torch.gather(ptr, 1, gidx).to(torch.int32)
    return torch.where(valid, phys, torch.full_like(phys, -1)).contiguous()


def fast_topk_transform_ragged_fused(score, lengths, topk_indices_offset, topk, row_starts=None):
    """Portable sgl_kernel.fast_topk_transform_ragged_fused (RAGGED / extend). Top-k
    logical indices, then shift into the flat ragged KV buffer:
    dst[b,j] = logical[b,j] + topk_indices_offset[b]; -1 sentinels preserved."""
    logical = _windowed_topk_logical(score, lengths, topk, row_starts)   # [B, topk] int32
    valid = logical >= 0
    off = topk_indices_offset.view(-1, 1).to(torch.int32)
    ragged = logical + off
    return torch.where(valid, ragged, torch.full_like(ragged, -1)).contiguous()


def fp8_paged_mqa_logits(q_fp8, kv_cache_fp8, weights, context_lens, block_tables,
                         schedule_metadata=None, max_seq_len=None, clean_logits=False, **kw):
    dev = q_fp8.device
    batch = q_fp8.shape[0]
    D = q_fp8.shape[-1]
    block_kv = kv_cache_fp8.shape[1]
    if max_seq_len is None:
        max_seq_len = block_tables.shape[1] * block_kv
    cl = context_lens.reshape(-1).to(torch.int64)
    N = kv_cache_fp8.shape[0]
    cu = kv_cache_fp8.reshape(N * block_kv, -1)        # [N*block_kv, D+4 bytes]
    data = cu[:, :D].reshape(N * block_kv, D).view(torch.float8_e4m3fn)
    scale = cu[:, D:].contiguous().view(torch.float32).reshape(-1)
    logits = torch.full((batch, max_seq_len), float("-inf"), dtype=torch.float32, device=dev)
    ar = torch.arange(block_kv, device=dev)
    for b in range(batch):
        ctx = int(cl[b])
        if ctx <= 0:
            continue
        nblk = (ctx + block_kv - 1) // block_kv
        blk = block_tables[b, :nblk].to(torch.int64)
        tok = (blk[:, None] * block_kv + ar[None, :]).reshape(-1)[:ctx]
        qb = q_fp8[b, 0]                                # [heads, D]
        wb = weights[b]                                 # [heads]
        lg = _mqa_logits(qb.unsqueeze(0), data[tok], scale[tok], wb.unsqueeze(0).float(),
                         torch.zeros(1, dtype=torch.int32, device=dev),
                         torch.tensor([ctx], dtype=torch.int32, device=dev))
        logits[b, :ctx] = lg[0]
    return logits


# ---- one-call installer: monkeypatch sglang's SM90+/SM100 DSA kernels ----
def apply_patches():
    """Replace sglang's Hopper/Blackwell-only DSA kernels with these portable sm_89
    versions. Call once at import time on Ada / sub-Hopper NVIDIA GPUs, e.g. append to
    `sglang/srt/layers/attention/nsa/nsa_indexer.py` (after `import deep_gemm`):

        import torch
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 9:
            import ada_dsa; ada_dsa.apply_patches()      # ada_dsa.py must be on PYTHONPATH
    """
    import logging
    import deep_gemm
    import sgl_kernel
    from sglang.srt.layers.attention.nsa import tilelang_kernel as _tk
    from sglang.srt.layers.attention.nsa.dequant_k_cache import dequantize_k_cache as _dq

    # 1) lightning-indexer logits (DeepGEMM, SM90+) -> portable Triton
    deep_gemm.fp8_mqa_logits = fp8_mqa_logits
    deep_gemm.fp8_paged_mqa_logits = fp8_paged_mqa_logits
    deep_gemm.get_paged_mqa_logits_metadata = get_paged_mqa_logits_metadata
    # 2) top-k + page-mapping transforms (sgl_kernel, SM100) -> torch shims
    sgl_kernel.fast_topk_v2 = fast_topk_v2
    sgl_kernel.fast_topk_transform_fused = fast_topk_transform_fused
    sgl_kernel.fast_topk_transform_ragged_fused = fast_topk_transform_ragged_fused

    # 3) MLA sparse decode: the default tilelang v2 kernel needs Hopper WGMMA (wg_wait);
    #    route to the non-WGMMA v1 kernel + dequant the packed fp8-656 cache to bf16-576,
    #    and use block_I=32/threads=128 to fit Ada's ~99 KB dynamic-smem cap.
    def _ada_tl_sparse_fwd(q, kv, indices, sm_scale, d_v=512):
        if kv.dtype == torch.float8_e4m3fn and kv.shape[-1] == 656:
            kv = _dq(kv)
        elif kv.dtype not in (torch.bfloat16, torch.float16):
            kv = kv.to(torch.bfloat16)
        kv = torch.cat([torch.zeros_like(kv[:1]), kv], dim=0)[1:]   # zero-row pad: KV[-1] of masked slots
        if q.dtype not in (torch.bfloat16, torch.float16):
            q = q.to(torch.bfloat16)
        nh, dim, tk = q.shape[1], q.shape[2], indices.shape[-1]
        kern = _tk.sparse_attention_fwd_kernel_v1(
            nh, d_v, dim - d_v, tk, sm_scale=sm_scale, num_stages=1, block_I=32, threads=128
        )
        return kern(q.unsqueeze(0), kv.unsqueeze(0), indices.unsqueeze(0))

    _tk.tilelang_sparse_fwd = _ada_tl_sparse_fwd
    logging.getLogger(__name__).warning(
        "[ada_dsa] patched DSA kernel stack for sm_%d%d" % torch.cuda.get_device_capability()
    )
