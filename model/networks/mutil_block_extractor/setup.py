#!/usr/bin/env python3
import os
import torch

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++14']

nvcc_args = [
    '-gencode', 'arch=compute_60,code=sm_60'
]

setup(
    name='mutil_block_extractor_cuda',
    ext_modules=[
        CUDAExtension('mutil_block_extractor_cuda', [
            'mutil_block_extractor_cuda.cc',
            'mutil_block_extractor_kernel.cu'
        ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    })

