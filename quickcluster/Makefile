
# x86_64-w64-mingw32-g++ for windows

CC=clang++
HEADERS=-Iinclude
CXX=-std=c++17
WARNINGS=-Wall -Wextra
OPTIMIZE=-O3

make:
	${CC} ${HEADERS} ${CXX} ${WARNINGS} src/quickcluster/kmeans/kmeans.cc src/quickcluster/linearalgebra/array.cc src/quickcluster/linearalgebra/distance.cc src/quickcluster/benchmark/stopwatch.cc src/quickcluster/datasets/clusters.cc ${OPTIMIZE} -fPIC -shared -o lib/libcluster.dylib

main:
	${CC} ${HEADERS} src/main.cc ${CXX} ${WARNINGS} -Llib -lcluster -Wl,-rpath,lib ${OPTIMIZE} -o main.o

run:
	./build/main.o