#!/usr/bin/env python3
"""Apply the two small sglang edits needed to run GLM-5.2's DSA on Ada (sm_89).

  1. Call `ada_dsa.apply_patches()` from `nsa_indexer.py` at import time on sub-Hopper
     GPUs (swaps the SM90+/SM100 DSA kernels for the portable ones in ada_dsa.py).
  2. Guard `configure_deep_gemm_num_sms` so CUDA-graph capture doesn't touch an
     unimportable `deep_gemm` (needed only for the ~4x CUDA-graph speedup).

Idempotent. Run once after installing sglang, with `ada_dsa.py` on PYTHONPATH:

    python apply_sglang_patches.py

Originals are backed up with a `.orig` suffix on first run.
"""
import importlib.util
import os
import shutil
import sys


def _sglang_root():
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.origin:
        sys.exit("sglang is not importable in this environment")
    return os.path.dirname(spec.origin)


def _backup(path):
    if not os.path.exists(path + ".orig"):
        shutil.copy2(path, path + ".orig")


def patch_nsa_indexer(root):
    f = os.path.join(root, "srt/layers/attention/nsa/nsa_indexer.py")
    src = open(f).read()
    if "ada_dsa.apply_patches()" in src:
        print("[skip] nsa_indexer.py already patched")
        return
    _backup(f)
    guard = (
        "\n\n# --- ada_dsa: portable DSA kernels for NVIDIA sm < 90 (Ada / RTX 4090) ---\n"
        "import torch as _ada_torch\n"
        "if _ada_torch.cuda.is_available() and _ada_torch.cuda.get_device_capability()[0] < 9:\n"
        "    import ada_dsa as _ada_dsa_mod\n"
        "    _ada_dsa_mod.apply_patches()\n"
    )
    open(f, "a").write(guard)
    print("[ok]   patched nsa_indexer.py (calls ada_dsa.apply_patches)")


def patch_deep_gemm_wrapper(root):
    f = os.path.join(root, "srt/layers/deep_gemm_wrapper/entrypoint.py")
    if not os.path.exists(f):
        print("[skip] deep_gemm_wrapper/entrypoint.py not present (older sglang); CUDA-graph guard not needed")
        return
    src = open(f).read()
    old = "def configure_deep_gemm_num_sms(num_sms):\n    if num_sms is None:\n"
    new = "def configure_deep_gemm_num_sms(num_sms):\n    if num_sms is None or 'deep_gemm' not in globals():\n"
    if new in src:
        print("[skip] deep_gemm_wrapper/entrypoint.py already patched")
        return
    if old not in src:
        print("[warn] entrypoint.py anchor not found (sglang version drift); apply the guard manually, see TECHNICAL.md")
        return
    _backup(f)
    open(f, "w").write(src.replace(old, new, 1))
    print("[ok]   patched deep_gemm_wrapper/entrypoint.py (CUDA-graph deep_gemm guard)")


if __name__ == "__main__":
    root = _sglang_root()
    print("sglang:", root)
    patch_nsa_indexer(root)
    patch_deep_gemm_wrapper(root)
    print("done. Make sure ada_dsa.py is importable (on PYTHONPATH).")
