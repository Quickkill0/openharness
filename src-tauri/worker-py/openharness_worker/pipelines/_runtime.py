"""Shared helpers for the heavy pipelines: device/dtype selection, VRAM
snapshots, and seeding.

Everything that imports torch stays inside functions — importing this
module must remain dependency-free so stdlib-only tests can load pipeline
modules directly.
"""

from __future__ import annotations

import secrets


def opt(params: dict, key: str, default, cast):
    """`params[key]` cast, or `default` when the key is absent/None.

    Unlike `params.get(key) or default`, an explicit falsy value (0, "")
    is kept and validated downstream rather than silently replaced.
    """
    v = params.get(key)
    return cast(v) if v is not None else default


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def device() -> str:
    return "cuda" if cuda_available() else "cpu"


def vram_used_mb() -> int | None:
    """Allocated CUDA memory in MiB, or None on CPU."""
    if not cuda_available():
        return None
    import torch

    return int(torch.cuda.memory_allocated() / (1024 * 1024))


def resolve_seed(raw) -> int:
    """A user-provided seed, or a fresh random one (always returned so the
    UI can show + reuse it)."""
    if raw is not None:
        return int(raw) & 0xFFFFFFFF
    return secrets.randbelow(2**32)


def make_generator(seed: int):
    import torch

    return torch.Generator(device=device()).manual_seed(seed)


def has_fp16_variant(path) -> bool:
    """Does a diffusers model dir ship `.fp16.` variant safetensors?

    Trimmed repo downloads keep only the fp16 copies, so `from_pretrained`
    must be pointed at the variant or it will look for the (absent)
    full-precision filenames. Diffusers falls back per-component with a
    warning when some component lacks the variant, so passing it when any
    fp16 file exists is safe.
    """
    from pathlib import Path

    p = Path(path)
    pats = ("*/*.fp16.safetensors", "*/*.fp16-*.safetensors")
    return any(next(p.glob(g), None) is not None for g in pats)


def free_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
