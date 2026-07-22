cmake_minimum_required(VERSION 3.22)
project(fast_alltoall
    VERSION 0.1
    DESCRIPTION "FAST NVSHMEM all-to-all reference implementation"
    LANGUAGES CXX
)

option(BUILD_FAST_EXTENSION "Build the FAST Torch/NVSHMEM extension" ON)
option(BUILD_FAST_SCHEDULER_BENCHMARK "Build the FAST scheduler-only CUDA benchmark" OFF)

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

if(BUILD_FAST_EXTENSION OR BUILD_FAST_SCHEDULER_BENCHMARK)
    enable_language(CUDA)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    find_package(CUDAToolkit REQUIRED)
endif()

if(BUILD_FAST_EXTENSION)
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

    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(Torch REQUIRED)
    add_library(torch_imported INTERFACE)
    add_library(torch::py_limited ALIAS torch_imported)
    target_include_directories(torch_imported SYSTEM INTERFACE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(torch_imported INTERFACE
        c10 torch torch_cpu c10_cuda torch_cuda CUDA::cudart
    )

    add_compile_options(-Wno-deprecated-declarations)
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)
    add_compile_definitions(Py_LIMITED_API=0x03090000)

    add_library(flash SHARED
        alltoall_nvshmem.cpp
        src/fast_alltoall/alltoall_algorithm.cc
        src/fast_alltoall/alltoall_global_scheduler.cc
        src/fast_alltoall/alltoall_local_scheduler.cc
        src/fast_alltoall/alltoall_matrix.cc
        src/fast_alltoall/flash_alltoall_nvshmem.cu
    )
    target_link_libraries(flash PUBLIC
        torch::py_limited
        Python::Module
        CUDA::cuda_driver
        CUDA::cudart
        nvshmem::nvshmem_host
        nvshmem::nvshmem_device
    )
    set_target_properties(flash PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

if(BUILD_FAST_SCHEDULER_BENCHMARK)
    add_executable(fast_scheduler_benchmark
        tests/fast_scheduler_benchmark.cc
        src/fast_alltoall/alltoall_algorithm.cc
        src/fast_alltoall/alltoall_global_scheduler.cc
        src/fast_alltoall/alltoall_local_scheduler.cc
        src/fast_alltoall/alltoall_matrix.cc
    )
    target_link_libraries(fast_scheduler_benchmark PRIVATE CUDA::cudart)
    target_compile_definitions(fast_scheduler_benchmark PRIVATE FAST_SCHEDULER_QUIET)
endif()
