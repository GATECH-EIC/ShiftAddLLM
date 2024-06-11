from setuptools import setup, Extension
from torch.utils import cpp_extension
import os
FORCE_SINGLE_THREAD = os.getenv("FLASH_ATTENTION_FORCE_SINGLE_THREAD", "FALSE") == "TRUE"

def append_nvcc_threads(nvcc_extra_args):
    if not FORCE_SINGLE_THREAD:
        return nvcc_extra_args + ["--threads", "4"]
    return nvcc_extra_args

setup(
    name='lutgemm',
    ext_modules=[cpp_extension.CUDAExtension(
        'lutgemm', ['lutgemm_cuda.cpp', 'lutgemm_cuda_kernel.cu'],

            extra_compile_args={
                "nvcc": append_nvcc_threads(
                    [
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "--ptxas-options=-v",
                        "--ptxas-options=-O2",
                        "-lineinfo",
                    ]
            )},


    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
