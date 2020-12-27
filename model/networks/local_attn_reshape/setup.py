#!/usr/bin/env python3
import os
import torch

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++14']

nvcc_args = [
    #'-gencode', 'arch=compute_50,code=sm_50',
    #'-gencode', 'arch=compute_52,code=sm_52',
    '-gencode', 'arch=compute_60,code=sm_60',
    # '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_75,code=sm_75',
    '-gencode', 'arch=compute_75,code=compute_75'
]

setup(
    name='local_attn_reshape_cuda',
    ext_modules=[
        CUDAExtension('local_attn_reshape_cuda', [
            'local_attn_reshape_cuda.cc',
            'local_attn_reshape_kernel.cu'
        ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    })

