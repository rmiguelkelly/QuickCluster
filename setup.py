import numpy as np
from setuptools import Extension, setup

package_name = 'quickcluster'

# Bindings for the low level things
def create_extension():
    return Extension(
        name="quickclusterinternal",
        include_dirs=[
            'include',
            np.get_include()
        ],
        sources=[ 
            'src/module/module.cc',
        ],
        library_dirs=[ '.' ],
        extra_link_args=[ '-lcluster' ],
        extra_compile_args=[ '-std=c++17' ]
)

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
        create_extension() 
    ],
)