cmake_minimum_required(VERSION 3.22)
project(dlb_alltoall
    VERSION 0.1
    DESCRIPTION "DLB source-local NVSHMEM all-to-all runtime"
    LANGUAGES CXX
)

option(BUILD_DLB_SCHEDULER_TESTS "Build DLB host scheduler tests" OFF)
option(BUILD_DLB_BENCHMARKS "Build DLB control-plane benchmarks" OFF)
option(BUILD_DLB_NVSHMEM_RUNTIME "Build the DLB multi-peer NVSHMEM/RDMA Rail backend" ON)
option(BUILD_DLB_NVLINK_RUNTIME "Build the DLB CUDA-IPC/NVLink staging runtime" ON)
option(BUILD_DLB_ALLTOALL_RUNTIME "Build the integrated DLB host/CUDA all-to-all runtime" ON)
option(BUILD_DLB_TORCH_EXTENSION "Build the DLB Torch/NVSHMEM extension" OFF)

set(NVSHMEM_ROOT "" CACHE PATH "Optional NVSHMEM installation root")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 90a CACHE STRING "CUDA architecture to target (for example: 90a)")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/include)

if(BUILD_DLB_SCHEDULER_TESTS OR BUILD_DLB_BENCHMARKS)
    add_library(dlb_scheduler STATIC
        src/dlb_alltoall/dlb_scheduler.cc
    )
    target_include_directories(dlb_scheduler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
endif()

if(BUILD_DLB_NVSHMEM_RUNTIME OR BUILD_DLB_ALLTOALL_RUNTIME OR BUILD_DLB_TORCH_EXTENSION)
    add_library(dlb_rail_receive_plan STATIC
        src/dlb_alltoall/dlb_rail_receive_plan.cc
    )
    target_include_directories(dlb_rail_receive_plan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
endif()

if(BUILD_DLB_SCHEDULER_TESTS)
    enable_testing()
    add_executable(test_dlb_scheduler tests/test_dlb_scheduler.cc)
    target_link_libraries(test_dlb_scheduler PRIVATE dlb_scheduler)
    add_test(NAME test_dlb_scheduler COMMAND test_dlb_scheduler)
endif()

if(BUILD_DLB_BENCHMARKS)
    find_package(Threads REQUIRED)
    add_executable(bench_dlb_scheduler benchmarks/bench_dlb_scheduler.cc)
    target_link_libraries(bench_dlb_scheduler PRIVATE dlb_scheduler Threads::Threads)
endif()

if(BUILD_DLB_TORCH_EXTENSION OR BUILD_DLB_NVSHMEM_RUNTIME OR
   BUILD_DLB_NVLINK_RUNTIME OR BUILD_DLB_ALLTOALL_RUNTIME)
    enable_language(CUDA)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    find_package(CUDAToolkit REQUIRED)
endif()

if(BUILD_DLB_TORCH_EXTENSION OR BUILD_DLB_NVSHMEM_RUNTIME OR BUILD_DLB_ALLTOALL_RUNTIME)
    find_package(NVSHMEM QUIET CONFIG HINTS ${NVSHMEM_ROOT})
    if(NOT TARGET nvshmem::nvshmem_host OR NOT TARGET nvshmem::nvshmem_device)
        find_path(NVSHMEM_INCLUDE_DIR nvshmem.h HINTS ${NVSHMEM_ROOT}/include REQUIRED)
        find_file(NVSHMEM_HOST_LIBRARY NAMES libnvshmem_host.so libnvshmem_host.so.3
            HINTS ${NVSHMEM_ROOT}/lib REQUIRED)
        find_file(NVSHMEM_DEVICE_LIBRARY NAMES libnvshmem_device.a
            HINTS ${NVSHMEM_ROOT}/lib REQUIRED)
        add_library(nvshmem::nvshmem_host UNKNOWN IMPORTED)
        set_target_properties(nvshmem::nvshmem_host PROPERTIES
            IMPORTED_LOCATION "${NVSHMEM_HOST_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${NVSHMEM_INCLUDE_DIR}"
        )
        add_library(nvshmem::nvshmem_device STATIC IMPORTED)
        set_target_properties(nvshmem::nvshmem_device PROPERTIES
            IMPORTED_LOCATION "${NVSHMEM_DEVICE_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${NVSHMEM_INCLUDE_DIR}"
        )
    endif()
endif()

if(BUILD_DLB_NVSHMEM_RUNTIME)
    add_library(dlb_nvshmem_runtime STATIC
        src/dlb_alltoall/dlb_rail_transport.cu
    )
    target_include_directories(dlb_nvshmem_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
    target_link_libraries(dlb_nvshmem_runtime PUBLIC
        dlb_rail_receive_plan
        CUDA::cudart
        nvshmem::nvshmem_host
        nvshmem::nvshmem_device
    )
    set_target_properties(dlb_nvshmem_runtime PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

if(BUILD_DLB_NVLINK_RUNTIME)
    add_library(dlb_nvlink_runtime STATIC
        src/dlb_alltoall/dlb_nvlink_runtime.cu
    )
    target_include_directories(dlb_nvlink_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
    target_link_libraries(dlb_nvlink_runtime PUBLIC
        CUDA::cudart
    )
    set_target_properties(dlb_nvlink_runtime PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

if(BUILD_DLB_ALLTOALL_RUNTIME)
    if(NOT BUILD_DLB_NVSHMEM_RUNTIME OR NOT BUILD_DLB_NVLINK_RUNTIME)
        message(FATAL_ERROR "BUILD_DLB_ALLTOALL_RUNTIME requires both DLB CUDA runtime libraries")
    endif()
    add_library(dlb_nvshmem_rdma_backend STATIC
        src/dlb_alltoall/dlb_alltoall_nvshmem.cu
    )
    target_include_directories(dlb_nvshmem_rdma_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
    target_link_libraries(dlb_nvshmem_rdma_backend PUBLIC
        dlb_nvlink_runtime
        dlb_nvshmem_runtime
        CUDA::cudart
        nvshmem::nvshmem_host
        nvshmem::nvshmem_device
    )
    set_target_properties(dlb_nvshmem_rdma_backend PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_library(dlb_alltoall_runtime ALIAS dlb_nvshmem_rdma_backend)
endif()

if(BUILD_DLB_TORCH_EXTENSION)
    if(NOT BUILD_DLB_ALLTOALL_RUNTIME)
        message(FATAL_ERROR "BUILD_DLB_TORCH_EXTENSION requires BUILD_DLB_ALLTOALL_RUNTIME")
    endif()
    find_package(Torch REQUIRED)

    add_library(dlb_torch INTERFACE)
    target_include_directories(dlb_torch SYSTEM INTERFACE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(dlb_torch INTERFACE
        c10 torch torch_cpu c10_cuda torch_cuda CUDA::cudart
    )

    add_library(dlb_nvshmem SHARED
        dlb_nvshmem_binding.cpp
        src/dlb_alltoall/dlb_moe_runtime.cu
    )
    target_compile_options(dlb_nvshmem PRIVATE -Wno-deprecated-declarations)
    target_compile_definitions(dlb_nvshmem PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
    target_link_libraries(dlb_nvshmem PUBLIC
        dlb_torch
        dlb_nvshmem_rdma_backend
        CUDA::cuda_driver
        CUDA::cudart
        nvshmem::nvshmem_host
        nvshmem::nvshmem_device
    )
    set_target_properties(dlb_nvshmem PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RUNTIME_LIBRARY Shared
    )
endif()
