

#include <linearalgebra/array.h>


bool test_array_equality_1() {

    Array<float> a1 = { 1, 2, 3, 4 };
    Array<float> a2 = { 1, 2, 3, 4 };

    return a1 == a2;
}

bool test_array_equality_2() {

    Array<float> a1 = { 1, 2, 3, 4 };
    Array<float> a2 = { 1, 2, 3, 4 };

    a1.resize(2, 2);
    a2.resize(2, 2);

    return a1 == a2;
}

bool test_array_equality_3() {

    Array<float> a1 = { 1, 2, 3, 4 };
    Array<float> a2 = { 1, 2 };

    return !(a1 == a2);
}

bool test_array_transposition() {

    Array<float> a1 = { 1, 2, 3, 4, 5, 6 };

    a1.resize(3, 2);

    // 1 2
    // 3 4
    // 5 6

    Array<float> a1_t = { 1, 3, 5, 2, 4, 6 };
    a1_t.resize(2, 3);

    return a1.transposed() == a1_t;
}