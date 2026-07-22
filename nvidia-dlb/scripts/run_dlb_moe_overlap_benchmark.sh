#!/usr/bin/env bash

set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "${script_dir}/.." && pwd)
python_bin=${DLB_PYTHON:-python3}
timestamp=$(date +%Y%m%d-%H%M%S)
suffix=$(printf '%04x' "$((RANDOM & 65535))")
run_dir="${project_root}/log/${timestamp}-${suffix}"
mkdir -p "${run_dir}"

command=(
    "${python_bin}" -m torch.distributed.run
    --standalone
    --nproc-per-node=8
    "${project_root}/benchmarks/bench_dlb_moe_overlap.py"
    --result-json "${run_dir}/result.json"
)
command+=("$@")

printf '%q ' "${command[@]}" > "${run_dir}/command.txt"
printf '\n' >> "${run_dir}/command.txt"
{
    printf 'run_id=%s\n' "$(basename "${run_dir}")"
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'working_directory=%s\n' "${project_root}"
    "${python_bin}" -c 'import torch; print(f"torch={torch.__version__}"); print(f"cuda={torch.version.cuda}"); print(f"gpu_count={torch.cuda.device_count()}"); print(f"capability={torch.cuda.get_device_capability(0)}")'
} > "${run_dir}/environment.txt" 2>&1

printf 'DLB overlap benchmark log: %s\n' "${run_dir}"
set +e
env NVSHMEM_IB_ENABLE_IBGDA="${NVSHMEM_IB_ENABLE_IBGDA:-true}" \
    NVSHMEM_DEBUG="${NVSHMEM_DEBUG:-WARN}" \
    "${command[@]}" 2>&1 | tee "${run_dir}/stdout.log"
exit_code=${PIPESTATUS[0]}
set -e

{
    if [[ ${exit_code} -eq 0 ]]; then
        printf 'status=passed\n'
    else
        printf 'status=failed\n'
    fi
    printf 'exit_code=%d\n' "${exit_code}"
    printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
} > "${run_dir}/status.txt"

printf 'DLB overlap benchmark status: %s (exit code %d)\n' \
    "$([[ ${exit_code} -eq 0 ]] && printf passed || printf failed)" "${exit_code}"
exit "${exit_code}"
