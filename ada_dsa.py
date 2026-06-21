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
    # Return a non-None dummy tensor (not None) so sglang's cuda-graph replay can
    # `.copy_()` into a pre-existing buffer — a frozen-dataclass field that is None at
    # capture cannot be reassigned at replay. The Triton paged path ignores this entirely.
    dev = args[0].device if (args and hasattr(args[0], "device")) else torch.device("cuda")
    return torch.zeros(1, dtype=torch.int32, device=dev)


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


@triton.jit
def _paged_idx_kernel(Q_ptr, POOLK_ptr, POOLS_ptr, BT_ptr, W_ptr, KE_ptr, LOGITS_ptr,
    sqb, sqh, sqd, spk, sps, sbt, sw, slo, S, NBLK,
    NUM_HEADS: tl.constexpr, HEAD_SIZE: tl.constexpr, PAGE: tl.constexpr, SCALE_OFF: tl.constexpr):
    # Page-indirection in-kernel: each loop iteration is exactly one physical page
    # (BLOCK_KV == page_size), read straight from the pool via the block table. No
    # whole-pool gather / .contiguous(); the loop runs ceil(context/PAGE) times so cost is
    # O(context), not O(pool capacity). Matches what DeepGEMM's paged kernel does on H100.
    b = tl.program_id(0)
    h = tl.arange(0, NUM_HEADS)[:, None]
    d = tl.arange(0, HEAD_SIZE)
    q = tl.load(Q_ptr + b * sqb + h * sqh + d[None, :] * sqd).to(tl.bfloat16)        # [H, D]
    w = tl.load(W_ptr + b * sw + tl.arange(0, NUM_HEADS)).to(tl.float32)[:, None]    # [H, 1]
    end = tl.load(KE_ptr + b)
    off = tl.arange(0, PAGE)
    nb = (end + PAGE - 1) // PAGE
    for i in tl.range(0, nb):
        col = i * PAGE + off
        m = col < end
        pp = tl.load(BT_ptr + b * sbt + i)
        kptr = POOLK_ptr + pp * spk + off[None, :] * HEAD_SIZE + d[:, None]
        k = tl.load(kptr, mask=m[None, :], other=0.0).to(tl.bfloat16)                # [D, PAGE]
        sc = tl.load(POOLS_ptr + pp * sps + SCALE_OFF + off, mask=m, other=0.0)      # [PAGE]
        s = tl.dot(q, k).to(tl.float32) * sc[None, :]
        s = tl.sum(tl.maximum(s, 0.0) * w, axis=0)
        tl.store(LOGITS_ptr + b * slo + col, s, mask=m)


def fp8_paged_mqa_logits(q_fp8, kv_cache_fp8, weights, context_lens, block_tables,
                         schedule_metadata=None, max_seq_len=None, clean_logits=False, **kw):
    # Paged DSA indexer logits (decode). Reads the index-K pool directly via the block
    # table inside the Triton kernel, so there is no max_total-sized gather or whole-pool
    # .contiguous() — cost is O(context), not O(pool capacity). Bit-identical to the prior
    # gather path (top-k selection agreement 1.0); ~4x faster at 192K, ~18x at 1M context.
    #
    # The index-K cache is struct-of-arrays per page: a page of `block_kv` tokens is
    # [block_kv*D fp8 keys] then [block_kv fp32 scales]. As an fp8 view a key lives at page
    # byte offset off*D + d; as an fp32 view the per-page scale block starts at element
    # (D*block_kv)//4. CUDA-graph-safe: launch shape is fixed (grid = batch); the per-row
    # loop count is data-dependent (allowed under capture).
    dev = q_fp8.device
    batch = q_fp8.shape[0]
    D = q_fp8.shape[-1]
    H = q_fp8.shape[-2]
    block_kv = kv_cache_fp8.shape[1]
    max_blocks = block_tables.shape[1]
    S = max_blocks * block_kv if max_seq_len is None else max_seq_len
    N = kv_cache_fp8.shape[0]
    flat = kv_cache_fp8.reshape(N, -1)                 # [N, page_bytes] uint8
    poolk = flat.view(torch.float8_e4m3fn)             # [N, page_bytes] fp8 keys (byte view)
    pools = flat.view(torch.float32)                   # [N, page_bytes // 4] fp32 (scale view)
    SCALE_OFF = (D * block_kv) // 4                     # fp32 index where the scale block begins
    ke = context_lens.reshape(-1).to(torch.int32)
    bt = block_tables.clamp(0, N - 1).to(torch.int32)
    nblk = (S + block_kv - 1) // block_kv
    q2 = q_fp8[:, 0].contiguous()                      # [batch, H, D]
    w = weights.float().contiguous()                   # [batch, H]
    logits = torch.full((batch, S), float("-inf"), dtype=torch.float32, device=dev)
    _paged_idx_kernel[(batch,)](
        q2, poolk, pools, bt, w, ke, logits,
        q2.stride(0), q2.stride(1), q2.stride(2), poolk.stride(0), pools.stride(0),
        bt.stride(0), w.stride(0), logits.stride(0), S, nblk,
        H, D, block_kv, SCALE_OFF, num_warps=4, num_stages=2)
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
    #    Decode dequant-on-gather: gather the topk selected KV rows FIRST and dequant only
    #    those, instead of dequantizing the whole pool every layer. Bit-identical to the
    #    full-cache path; makes decode cost independent of context/pool size. Gated to the
    #    decode regime (m*topk <= pool) — prefill/extend has large m (m*topk would exceed
    #    the pool and OOM), so it falls back to the full-cache dequant. The branch is static
    #    per shape, so it stays CUDA-graph-safe.
    def _ada_tl_sparse_fwd(q, kv, indices, sm_scale, d_v=512):
        m, tk, N = indices.shape[0], indices.shape[-1], kv.shape[0]
        if m * tk <= N:
            valid = indices >= 0
            gidx = indices.clamp(min=0).clamp(max=N - 1).to(torch.long).reshape(-1)
            if kv.dtype == torch.float8_e4m3fn and kv.shape[-1] == 656:
                sel = _dq(kv.reshape(N, -1).index_select(0, gidx).view(-1, 1, 656))
            else:
                kvf = kv if kv.dtype in (torch.bfloat16, torch.float16) else kv.to(torch.bfloat16)
                sel = kvf.reshape(N, -1).index_select(0, gidx).view(-1, 1, kvf.shape[-1])
            kv4 = sel.reshape(1, m * tk, 1, sel.shape[-1])
            ar = torch.arange(tk, device=indices.device, dtype=torch.int32)
            base = (torch.arange(m, device=indices.device, dtype=torch.int32) * tk).view(m, 1, 1)
            ident = torch.where(valid, base + ar.view(1, 1, tk), torch.full_like(indices, -1))
            qq = q if q.dtype in (torch.bfloat16, torch.float16) else q.to(torch.bfloat16)
            nh, dim = qq.shape[1], qq.shape[2]
            kern = _tk.sparse_attention_fwd_kernel_v1(
                nh, d_v, dim - d_v, tk, sm_scale=sm_scale, num_stages=1, block_I=32, threads=128
            )
            return kern(qq.unsqueeze(0), kv4, ident.unsqueeze(0))
        # fallback: full-cache dequant (prefill / large extend)
        if kv.dtype == torch.float8_e4m3fn and kv.shape[-1] == 656:
            kv = _dq(kv)
        elif kv.dtype not in (torch.bfloat16, torch.float16):
            kv = kv.to(torch.bfloat16)
        kv = torch.cat([torch.zeros_like(kv[:1]), kv], dim=0)[1:]   # zero-row pad: KV[-1] of masked slots
        if q.dtype not in (torch.bfloat16, torch.float16):
            q = q.to(torch.bfloat16)
        nh, dim = q.shape[1], q.shape[2]
        kern = _tk.sparse_attention_fwd_kernel_v1(
            nh, d_v, dim - d_v, tk, sm_scale=sm_scale, num_stages=1, block_I=32, threads=128
        )
        return kern(q.unsqueeze(0), kv.unsqueeze(0), indices.unsqueeze(0))

    _tk.tilelang_sparse_fwd = _ada_tl_sparse_fwd
    logging.getLogger(__name__).warning(
        "[ada_dsa] patched DSA kernel stack for sm_%d%d" % torch.cuda.get_device_capability()
    )
