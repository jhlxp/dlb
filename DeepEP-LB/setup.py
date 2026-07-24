import os
import subprocess
import setuptools
import importlib

from pathlib import Path
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# Wheel specific: the wheels only include the soname of the host library `libnvshmem_host.so.X`
def get_nvshmem_host_lib_name(base_dir):
    path = Path(base_dir).joinpath('lib')
    for file in path.rglob('libnvshmem_host.so.*'):
        return file.name
    raise ModuleNotFoundError('libnvshmem_host.so not found')


if __name__ == '__main__':
    disable_nvshmem = False
    nvshmem_dir = os.getenv('NVSHMEM_DIR', None)
    nvshmem_host_lib = 'libnvshmem_host.so'
    if nvshmem_dir is None:
        try:
            nvshmem_dir = importlib.util.find_spec("nvidia.nvshmem").submodule_search_locations[0]
            nvshmem_host_lib = get_nvshmem_host_lib_name(nvshmem_dir)
            import nvidia.nvshmem as nvshmem
        except (ModuleNotFoundError, AttributeError, IndexError):
            print('Warning: `NVSHMEM_DIR` is not specified, and the NVSHMEM module is not installed. All internode and low-latency features are disabled\n')
            disable_nvshmem = True
    else:
        disable_nvshmem = False
        nvshmem_host_lib = get_nvshmem_host_lib_name(nvshmem_dir)

    if not disable_nvshmem:
        assert os.path.exists(nvshmem_dir), f'The specified NVSHMEM directory does not exist: {nvshmem_dir}'

    cxx_flags = ['-O3', '-Wno-deprecated-declarations', '-Wno-unused-variable',
                 '-Wno-sign-compare', '-Wno-reorder', '-Wno-attributes']
    nvcc_flags = ['-O3', '-Xcompiler', '-O3']
    num_nvl_peers = int(os.getenv('DEEPEP_NUM_NVL_PEERS', '8'))
    if num_nvl_peers not in (4, 8):
        raise ValueError('DEEPEP_NUM_NVL_PEERS must be 4 or 8')
    cxx_flags.append(f'-DNUM_MAX_NVL_PEERS={num_nvl_peers}')
    nvcc_flags.append(f'-DNUM_MAX_NVL_PEERS={num_nvl_peers}')
    sources = [
        'csrc/deep_ep.cpp',
        'csrc/kernels/runtime.cu',
        'csrc/kernels/layout.cu',
        'csrc/kernels/rail_lb.cu',
        'csrc/kernels/intranode.cu',
    ]
    include_dirs = ['csrc/']
    library_dirs = []
    nvcc_dlink = []
    extra_link_args = []

    # NVSHMEM flags
    if disable_nvshmem:
        cxx_flags.append('-DDISABLE_NVSHMEM')
        nvcc_flags.append('-DDISABLE_NVSHMEM')
    else:
        sources.extend(['csrc/kernels/internode.cu', 'csrc/kernels/internode_ll.cu'])
        include_dirs.extend([f'{nvshmem_dir}/include'])
        library_dirs.extend([f'{nvshmem_dir}/lib'])
        nvcc_dlink.extend(['-dlink', f'-L{nvshmem_dir}/lib', '-lnvshmem_device'])
        extra_link_args.extend([f'-l:{nvshmem_host_lib}', '-l:libnvshmem_device.a', f'-Wl,-rpath,{nvshmem_dir}/lib'])

    # DeepEP-LB targets Hopper only. Do not infer the architecture from a
    # possibly customized device name or inherit a stale build environment.
    if int(os.getenv('DISABLE_SM90_FEATURES', 0)):
        raise ValueError('DeepEP-LB requires SM90 features')
    os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'
    nvcc_flags.extend(['-rdc=true', '--ptxas-options=--register-usage-level=10'])

    # Disable aggressive PTX instructions
    if int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', '1')):
        cxx_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')
        nvcc_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')

    # Put them together
    extra_compile_args = {
        'cxx': cxx_flags,
        'nvcc': nvcc_flags,
    }
    if len(nvcc_dlink) > 0:
        extra_compile_args['nvcc_dlink'] = nvcc_dlink

    # Summary
    print(f'Build summary:')
    print(f' > Sources: {sources}')
    print(f' > Includes: {include_dirs}')
    print(f' > Libraries: {library_dirs}')
    print(f' > Compilation flags: {extra_compile_args}')
    print(f' > Link flags: {extra_link_args}')
    print(f' > Arch list: {os.environ["TORCH_CUDA_ARCH_LIST"]}')
    print(f' > GPUs per logical server: {num_nvl_peers}')
    print(f' > NVSHMEM path: {nvshmem_dir}')
    print()

    # noinspection PyBroadException
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(
            cmd,
            stderr=subprocess.DEVNULL,
        ).decode('ascii').rstrip()
    except Exception as _:
        revision = ''

    setuptools.setup(
        name='deep_ep',
        version='1.2.1' + revision,
        packages=setuptools.find_packages(
            include=['deep_ep']
        ),
        ext_modules=[
            CUDAExtension(
                name='deep_ep_cpp',
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                sources=sources,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
