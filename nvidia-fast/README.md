# NVIDIA FAST

This directory contains the original FAST reference implementation. It is a
standalone CMake project and has no source or build dependency on
`../nvidia-dlb`.

## Layout

- `alltoall_nvshmem.cpp`: Torch/NVSHMEM host binding and test entrypoint.
- `src/fast_alltoall/`: FAST scheduler and CUDA/NVSHMEM implementation.
- `src/include/fast_alltoall/`: FAST public and internal headers.
- `flash_utils.py`, `flash_tester.py`: Python loader and distributed test.
- `tests/fast_scheduler_benchmark.cc`: scheduler-only benchmark.

## Build

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/torch/share/cmake \
  -DNVSHMEM_ROOT=/path/to/nvshmem \
  -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build -j
```

The build writes `libflash.so` into this directory for `flash_utils.py`.

To build only the scheduler benchmark:

```bash
cmake -S . -B build-scheduler \
  -DBUILD_FAST_EXTENSION=OFF \
  -DBUILD_FAST_SCHEDULER_BENCHMARK=ON
cmake --build build-scheduler --target fast_scheduler_benchmark -j
```
