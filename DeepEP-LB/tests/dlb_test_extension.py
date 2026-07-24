"""Build and load test-only CUDA bindings for DLB validation."""

from __future__ import annotations

import os
from pathlib import Path

from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[1]


def load_dlb_test_extension():
    """Compile compact quota and CUDA-IPC/P2P tests for Hopper."""
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
    return load(
        name="deep_ep_lb_test_cpp",
        sources=[
            str(ROOT / "tests/csrc/dlb_core_test.cpp"),
            str(ROOT / "tests/csrc/dlb_core_test.cu"),
            str(ROOT / "tests/csrc/dlb_loopback_test.cu"),
            str(ROOT / "csrc/kernels/rail_lb.cu"),
        ],
        extra_include_paths=[
            str(ROOT / "csrc"),
        ],
        extra_cflags=[
            "-O3",
            "-DNUM_MAX_NVL_PEERS=4",
            "-DDISABLE_NVSHMEM",
        ],
        extra_cuda_cflags=[
            "-O3",
            "-DNUM_MAX_NVL_PEERS=4",
            "-DDISABLE_NVSHMEM",
        ],
        with_cuda=True,
        verbose=False,
    )
