
# x86_64-w64-mingw32-g++ for windows

CC=clang++
HEADERS=-Iinclude
CXX=-std=c++17
WARNINGS=-Wall -Wextra
OPTIMIZE=-O3

make:
	${CC} ${HEADERS} ${CXX} ${WARNINGS} src/quickcluster/kmeans/kmeans.cc src/quickcluster/linearalgebra/array.cc src/quickcluster/linearalgebra/distance.cc src/quickcluster/benchmark/stopwatch.cc src/quickcluster/datasets/clusters.cc ${OPTIMIZE} -fPIC -shared -o libcluster.dylib

main:
	${CC} ${HEADERS} ${CXX} ${WARNINGS} src/quickcluster/kmeans/kmeans.cc src/quickcluster/linearalgebra/array.cc src/quickcluster/linearalgebra/distance.cc src/quickcluster/benchmark/stopwatch.cc src/quickcluster/datasets/clusters.cc src/main.cc ${OPTIMIZE} -o build/main.o

run:
	./build/main.o