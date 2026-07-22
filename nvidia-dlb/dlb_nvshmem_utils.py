import os

import torch

_lib_path = os.path.join(os.path.dirname(__file__), "libdlb_nvshmem.so")
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
    """Construct a raw DLB transport with explicit production capacities."""
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
    return torch.ops.dlb_nvshmem.get_uniqueid()


def cuda_module_probe(device: int) -> torch.Tensor:
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
