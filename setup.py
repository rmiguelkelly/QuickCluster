import numpy as np
from setuptools import Extension, setup

import os
import platform

package_name = 'quickcluster'

extra_link_args = []

def relative_path(to):
    base = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(base, to)

if platform.system() == 'Darwin':
    extra_link_args.append('-Wl,-rpath,@loader_path/lib')

# Python package initiation
setup(
    name=package_name,
    author='Ronan M Kelly',
    author_email='thekelpez@gmail.com',
    description='A KMeans algorithm',
    version='1.0.1',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    packages=[
        'quickcluster'
    ],
    install_requires=[ 
        'numpy'
    ],
    ext_modules=[ 
        Extension(
            name="quickcluster._C",
            include_dirs=[
                'quickcluster/include',
                np.get_include()
            ],
            sources=[ 
                'quickcluster/src/module/module.cc',
            ],
            library_dirs=[ 'quickcluster/lib' ],
            libraries=[ 'cluster' ],
            extra_compile_args=[ '-std=c++17' ],
            extra_link_args=extra_link_args
        )
    ],
    package_data={
        'quickcluster': [
            'lib/*.dylib',
            'lib/*.so',
            'lib/*.dll',
            'include/**',
        ]
    }
)