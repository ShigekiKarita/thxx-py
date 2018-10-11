import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

from thxx import __version__

conda = os.getenv("CONDA_PREFIX")
if conda:
    inc = [conda + "/include"]
else:
    inc = []

setup(
    name='thxx',
    version=__version__,
    description='various C++ extensions for pytorch',
    url='https://github.com/ShigekiKarita/thxx',
    author='Shigeki Karita',
    author_email="shigekikarita@gmail.com",
    license='BSL-1.0',
    keywords='pytorch',
    ext_modules=[
        CppExtension(
            'thxx_autograd',
            ['autograd.cpp'],
            extra_compile_args=["-fopenmp"]
        ),
        CppExtension(
            'thxx_backend',
            ['backend.cpp'],
            include_dirs=inc,
            libraries=["cusolver", "cublas"]
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    packages=["thxx"],
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
