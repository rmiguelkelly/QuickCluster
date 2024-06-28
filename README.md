# QuickCluster 🔥
A KMeans algorithm implemented in C++ along with Pythoin bindings that runs on the CPU or the GPU. 

I wrote this project more as a proof of concept for some key learning points I wanted to understand.

This project is definitely still in beta so there may be some bugs and upgrades to add as time goes on.  


## Features

- Simple Python interface similar to SKLearn's KMeans
- Dynamic libraries for MacOS and Linux (Windows coming soon) for C++ integration
- Support for Apple's GPU for hardware acceleration (CUDA coming soon too)


## Building

Build the dynamic library from source

```bash
git clone https://github.com/rmiguelkelly/QuickCluster
cd QuickCluster/quickcluster
```

MacOS
```bash
make darwin
```

Linux
```bash
make linux
```

Windows and CUDA implementations
```bash
Coming soon!
```


## Examples
Some example code for running the program

Python (CPU) [Example](examples/kmeans-example-cpu.ipynb)

Python (GPU) [Example](examples/kmeans-example-gpu.ipynb)

C++ (CPU) [Example](examples/kmeans-example-cpu.cc)

C++ (GPU) [Example](examples/kmeans-example-gpu.cc)

    
## Python Installation

You must build the project first with the above steps

```bash
cd ..
pip install .
```

Note: you may have to edit the runtime paths using a tool like `install_name_tool` on MacOS for the dynamic library so that the Python module can link it properly


## Authors

- [@rmiguelkelly](https://www.github.com/rmiguelkelly)
