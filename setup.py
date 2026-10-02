"""Build the BiLoCo CUDA extension (biloco._C) for Blackwell B300 (sm_103a).

    git submodule update --init third_party/cutlass
    git -C third_party/cutlass apply ../../patches/cutlass_biloco.patch
    pip install -e . --no-build-isolation

Set BILOCO_CUDA_ARCH (default 103a) to build for another Blackwell part, e.g. 100a for B200.
"""
import os
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
CUTLASS = Path(os.getenv("CUTLASS_DIR", ROOT / "third_party" / "cutlass"))
HOOK = CUTLASS / "include" / "cutlass" / "gemm" / "kernel" / "binarysvd_side_hook.hpp"
if not HOOK.exists():
    raise RuntimeError(
        f"{HOOK} not found: initialize the CUTLASS submodule and apply patches/cutlass_biloco.patch "
        "(see the module docstring of setup.py)")

arch = os.getenv("BILOCO_CUDA_ARCH", "103a")
nvcc_flags = [
    "-O3", "-std=c++17", "-DNDEBUG",
    f"-gencode=arch=compute_{arch},code=sm_{arch}",
    "--expt-relaxed-constexpr", "--use_fast_math",
    "-Xcompiler", "-ffast-math", "-Xcompiler", "-funroll-loops", "-Xcompiler", "-finline-functions",
]

setup(
    name="biloco",
    version="0.1.0",
    description="BiLoCo: binary low-rank + NVFP4 residual decomposition and a fused Blackwell decode kernel",
    packages=find_packages(include=["biloco", "biloco.*"]),
    ext_modules=[
        CUDAExtension(
            name="biloco._C",
            sources=["csrc/bindings.cpp", "csrc/biloco_gemm.cu", "csrc/reference_ops.cu"],
            include_dirs=[
                str(ROOT / "csrc"),
                str(CUTLASS / "include"),
                str(CUTLASS / "tools" / "util" / "include"),
                str(CUTLASS / "examples" / "common"),
            ],
            extra_compile_args={"cxx": ["-O3", "-std=c++17"], "nvcc": nvcc_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
