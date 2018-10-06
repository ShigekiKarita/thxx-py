from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

from torch_autograd_solver import __version__

setup(
    name='torch_autograd_solver',
    version=__version__,
    description='autograd solver C++ implementation for pytorch',
    url='https://github.com/ShigekiKarita/pytorch-autograd-solver',
    author='Shigeki Karita',
    author_email="shigekikarita@gmail.com",
    license='BSL-1.0',
    keywords='pytorch',
    ext_modules=[
        CppExtension(
            'torch_autograd_solver_aten',
            ['torch_autograd_solver.cpp'],
            extra_compile_args=["-fopenmp"]
        )],
    cmdclass={'build_ext': BuildExtension},
    packages=["torch_autograd_solver"],
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
