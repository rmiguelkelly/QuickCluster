# ========================== #
#        - Makefile -        #
# ========================== #

# Common compiler arguments and params 
HEADERS=-Iinclude
CXX=-std=c++17
WARNINGS=-Wall -Wextra
OPTIMIZE=-O3


# Compiler options
CC=clang++
OPTIONS=${HEADERS} ${WARNINGS} ${OPTIMIZE} -fPIC


# Apple related build things
OBJC_FRAMEWORKS=-framework Foundation -framework CoreGraphics -framework Metal
XC=xcrun
METAL_PLATFORM=macosx


# CUDA related build things
NVCC=nvcc
NVCC_LIBS=-lcudart

# Build for MacOS
darwin: kmeans.o stopwatch.o clusters.o metal.o
	${CC} ${OPTIONS} ${CXX} ${OBJC_FRAMEWORKS} \
	build/kmeans.o build/array.o build/distance.o build/stopwatch.o build/clusters.o build/metal.o build/metaldevicemanager.o \
	-shared -o lib/libcluster.dylib

# Build for Linux
linux: kmeans.o stopwatch.o clusters.o
	${CC} ${OPTIONS} ${CXX} \
	build/kmeans.o build/array.o build/distance.o build/stopwatch.o build/clusters.o \
	-shared -o lib/libcluster.so

# Build for Linux + CUDA
linux-cuda: kmeans.o stopwatch.o clusters.o cuda.o compute_centroids.o
	${CC} ${OPTIONS} ${CXX} \
	build/kmeans.o build/array.o build/distance.o build/stopwatch.o build/clusters.o build/cuda.o build/compute_centroids.o \
	-shared -o lib/libcluster.so


# Platform independent logic for KMeans
kmeans.o: array.o distance.o
	@${CC} ${OPTIONS} ${CXX} src/quickcluster/kmeans/kmeans.cc -c -o build/kmeans.o

array.o:
	@${CC} ${OPTIONS} ${CXX} src/quickcluster/linearalgebra/array.cc -c -o build/array.o

distance.o:
	@${CC} ${OPTIONS} ${CXX} src/quickcluster/linearalgebra/distance.cc -c -o build/distance.o

stopwatch.o:
	@${CC} ${OPTIONS} ${CXX} src/quickcluster/benchmark/stopwatch.cc -c -o build/stopwatch.o

clusters.o:
	@${CC} ${OPTIONS} ${CXX} src/quickcluster/datasets/clusters.cc -c -o build/clusters.o


# Apple specific builds
metal.o: metaldevicemanager.o
	@${CC} ${OPTIONS} ${CXX} -fobjc-arc src/quickcluster/device/metal.mm -c -o build/metal.o

metaldevicemanager.o: compute_centroids.metallib
	@${CC} ${OPTIONS} ${CXX} -fobjc-arc src/quickcluster/device/MetalDeviceManager.mm -c -o build/metaldevicemanager.o

compute_centroids.metallib:
	@${XC} -sdk ${METAL_PLATFORM} metal -o build/cluster.ir  -c src/quickcluster/device/cluster.metal -Iinclude
	@${XC} -sdk ${METAL_PLATFORM} metallib -o lib/cluster.metallib build/cluster.ir


# CUDA specific builds
cuda.o: compute_centroids.o
	@${NVCC} ${HEADERS} ${OPTIMIZE} ${CXX} src/quickcluster/device/cuda.cu -c -o build/cuda.o

compute_centroids.o:
	@${NVCC} ${HEADERS} ${OPTIMIZE} ${CXX} src/quickcluster/device/compute_centroids.cu -c -o build/compute_centroids.o




# Auxilary commands for helpful stuff
main:
	${CC} ${HEADERS} ${CXX} ${WARNINGS} src/main.cc -Llib -lcluster -Wl,-rpath,lib ${OPTIMIZE} -o main.o

run:
	./build/main.o

clean: 
	rm -rf build/*
	rm -rf lib/*