"""Build/load the FlashInfer SM90 grouped-GEMM-only benchmark module.

FlashInfer 0.6.15 does not expose the two CUTLASS grouped GEMMs used by its
MXFP4 fused-MoE path as standalone Python calls.  The accompanying patch adds
an FFI entry that returns immediately after GEMM1 or GEMM2.  This loader
applies that patch only while compiling a uniquely named JIT module and then
restores every installed FlashInfer source file byte-for-byte.

The resulting module is benchmark-only: normal ``fused_moe_90`` AOT/JIT code
is never replaced, and no SGLang server, profiler, or CUDA graph is involved.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path

from filelock import FileLock


PATCH_PATH = (
    Path(__file__).resolve().parents[1]
    / "patches"
    / "flashinfer"
    / "0001-Add-pure-grouped-GEMM-profiling-entry.patch"
)
PATCHED_FILES = (
    Path("fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh"),
    Path("fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_binding.cu"),
    Path(
        "nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/"
        "moe_kernels.h"
    ),
)


def _flashinfer_version() -> str:
    for distribution in ("flashinfer-python", "flashinfer_python", "flashinfer"):
        try:
            return version(distribution)
        except Exception:
            pass
    return "unknown"


def _module_name() -> str:
    digest = hashlib.sha256(PATCH_PATH.read_bytes()).hexdigest()[:12]
    return f"fused_moe_grouped_gemm_only_90_{digest}"


def _make_spec(name: str):
    from flashinfer.jit.fused_moe import gen_cutlass_fused_moe_sm90_module

    spec = gen_cutlass_fused_moe_sm90_module()
    # A distinct name prevents both an AOT hit on fused_moe_90 and collision
    # with FlashInfer's production JIT cache.
    spec.name = name
    return spec


def _run_git_apply(csrc_dir: Path) -> None:
    command = [
        "git",
        "apply",
        "--unsafe-paths",
        "--whitespace=nowarn",
        str(PATCH_PATH),
    ]
    check = subprocess.run(
        command[:2] + ["--check"] + command[2:],
        cwd=csrc_dir.parent,
        text=True,
        capture_output=True,
    )
    if check.returncode:
        raise RuntimeError(
            "FlashInfer JIT sources do not match the supported 0.6.15.post1 "
            "tree; git apply --check failed:\n"
            + (check.stderr or check.stdout)
        )
    subprocess.run(command, cwd=csrc_dir.parent, check=True)


def load_grouped_gemm_only_module():
    """Return the patched SM90 FFI module, compiling it once if necessary."""
    installed_version = _flashinfer_version()
    if not installed_version.startswith("0.6.15"):
        raise RuntimeError(
            "The grouped-GEMM-only patch targets FlashInfer 0.6.15.x; found "
            f"{installed_version}."
        )
    if not PATCH_PATH.is_file():
        raise RuntimeError(f"missing FlashInfer source patch: {PATCH_PATH}")

    from flashinfer.jit import env as jit_env

    name = _module_name()
    lock_path = Path(tempfile.gettempdir()) / f"{name}.source.lock"
    with FileLock(str(lock_path), thread_local=False):
        # A hash-versioned cached library is known to contain this exact patch.
        # Loading it directly also avoids touching package sources on later runs.
        spec = _make_spec(name)
        if spec.jit_library_path.is_file():
            return spec.load(spec.jit_library_path)

        csrc_dir = Path(jit_env.FLASHINFER_CSRC_DIR).resolve()
        paths = [csrc_dir / relative for relative in PATCHED_FILES]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise RuntimeError(
                "FlashInfer wheel does not include required JIT sources: "
                + ", ".join(missing)
            )

        backups = {
            path: (path.read_bytes(), path.stat().st_atime_ns, path.stat().st_mtime_ns)
            for path in paths
        }
        binding = paths[1]
        already_patched = b"run_grouped_gemm_only" in binding.read_bytes()
        try:
            if not already_patched:
                _run_git_apply(csrc_dir)
            # Recreate the spec after patching so Ninja observes the patched
            # sources on the first build.  The unique module name is retained.
            spec = _make_spec(name)
            return spec.build_and_load()
        finally:
            if not already_patched:
                for path, (data, atime_ns, mtime_ns) in backups.items():
                    path.write_bytes(data)
                    os.utime(path, ns=(atime_ns, mtime_ns))


def main() -> None:
    module = load_grouped_gemm_only_module()
    print(f"Loaded {_module_name()} from {module!r}")


if __name__ == "__main__":
    main()
