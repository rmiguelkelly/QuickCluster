import numpy as np
from setuptools import Extension, setup
import platform
import os

# QuickCluster
package_name = 'quickcluster'

# Linker arguments
extra_link_args = []

platform_name = platform.system()

# Setup the necessary compiler linker settings for the dynamic library
if platform_name == 'Darwin':
    extra_link_args.append('-Wl,-rpath,@loader_path/lib')
elif platform_name == "Linux":
    extra_link_args.append('-Wl,-rpath,$ORIGIN/lib')
elif platform_name == "Windows":
    print("Not yet implemented!")
    pass
else:
    raise Exception("Unsupported operating system")

# Current directory
cwd = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(cwd, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


# Python package initiation
setup(
    name=package_name,
    author='Ronan M Kelly',
    author_email='thekelpez@gmail.com',
    description='A KMeans algorithm implemented in C++ and bridged to Python with GPU support',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='1.0.1',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    packages=[
        package_name
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
            'lib/*.metallib',
            'include/**',
        ]
    }
)