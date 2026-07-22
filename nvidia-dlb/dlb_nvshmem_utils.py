"""Loads the DLB Torch extension and exposes its low-level runtime objects."""

import os

import torch

_lib_path = os.path.join(os.path.dirname(__file__), "libdlb_nvshmem.so")
# Load both operator and custom-class registrations from the compiled extension.
torch.ops.load_library(_lib_path)
torch.classes.load_library(_lib_path)

_nvshmem_comm_t = torch.classes.dlb_classes.nvshmem_comm_t


def nvshmem_comm_t(
    rank: int,
    rails: int,
    world: int,
    device: int,
    uid: torch.Tensor,
    record_bytes: int,
    receive_slot_bytes: int,
    repair_slot_bytes: int,
    pipeline_depth: int,
    chunk_bytes: int,
    rail_send_capacity_bytes: int,
    num_comm_sms: int = 24,
    enable_profiling: bool = False,
    use_loopback_transport: bool = False,
):
    """Creates a raw DLB transport with explicit production capacities.

    Args:
        rank: Rank of the current process in the DLB process group.
        rails: Number of GPUs and Rail endpoints on each server.
        world: Total number of ranks in the DLB process group.
        device: CUDA device index owned by the current process.
        uid: NVSHMEM unique identifier shared by all ranks.
        record_bytes: Aligned size of one wire record in bytes.
        receive_slot_bytes: Capacity of one receive pipeline slot in bytes.
        repair_slot_bytes: Capacity of one destination-repair slot in bytes.
        pipeline_depth: Number of reusable communication pipeline slots.
        chunk_bytes: Maximum payload size of one transport chunk.
        rail_send_capacity_bytes: Per-rank Rail staging capacity in bytes.
        num_comm_sms: Number of SMs reserved for communication kernels.
        enable_profiling: Whether to record per-stage CUDA timings.
        use_loopback_transport: Whether to use the same-host test backend.

    Returns:
        The registered C++ DLB communication object.
    """
    return _nvshmem_comm_t(
        rank,
        rails,
        world,
        device,
        uid,
        record_bytes,
        receive_slot_bytes,
        repair_slot_bytes,
        pipeline_depth,
        chunk_bytes,
        num_comm_sms,
        rail_send_capacity_bytes,
        enable_profiling,
        use_loopback_transport,
    )


def get_nvshmem_init_id() -> torch.Tensor:
    """Returns a new NVSHMEM unique identifier as a CPU byte tensor."""
    return torch.ops.dlb_nvshmem.get_uniqueid()


def cuda_module_probe(device: int) -> torch.Tensor:
    """Verifies that the extension contains code for a CUDA device.

    Args:
        device: CUDA device index to probe.

    Returns:
        A CUDA tensor produced by the extension probe kernel.

    Raises:
        RuntimeError: If the extension was built for an incompatible GPU
            architecture or if the probe kernel fails.
    """
    try:
        return torch.ops.dlb_nvshmem.cuda_module_probe(device)
    except RuntimeError as error:
        if "named symbol not found" not in str(error):
            raise
        major, minor = torch.cuda.get_device_capability(device)
        raise RuntimeError(
            "DLB CUDA module architecture mismatch: device capability is "
            f"sm_{major}{minor}. Reconfigure in a clean build directory with "
            f"TORCH_CUDA_ARCH_LIST={major}.{minor} and "
            f"-DCMAKE_CUDA_ARCHITECTURES={major}{minor}."
        ) from error
