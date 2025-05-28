from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

sources = [
    "src/pybind.cpp",
    "src/cumc.cpp",
    "src/cudualmc.cu",
]

print("sources:", sources)

setup(
    name='diso',
    ext_modules=[
        CUDAExtension(
            name='diso._C',
            sources=sources,
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
