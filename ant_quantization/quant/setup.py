  
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='quant_cuda',
    ext_modules=[
        CUDAExtension('quant_cuda', [
            'quant.cpp',
            'quant_kernel.cu'],
            include_dirs=['includes'],
            ),
        
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
