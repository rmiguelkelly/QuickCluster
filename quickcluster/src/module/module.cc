
#define PY_SSIZE_T_CLEAN
#include <Python.h>

// C++ method bindings
#include <module/bindings.h>

// C++ KMeans interface
#include <quickcluster/kmeans/kmeans.h>
#include <quickcluster/linearalgebra/array.h>

// GPU stuff
#ifdef __APPLE__
#include <quickcluster/device/metal.h>
#endif

// Capsule object identifiers
#define KMEANS_POINTER_ID "kmeans.internal_handle"
#define DEVICE_HANDLE_ID "device.internal_handle"

// Called once python is done with the reference
void kmeans_destructor(PyObject* arg) {
    KMeans *ref = (KMeans*)PyCapsule_GetPointer(arg, KMEANS_POINTER_ID);
    delete ref;
}


// Called when the GPU handle is deallocated
void device_destructor(PyObject* arg) {

    // Get the handle for the device
    DeviceHandle handle = PyCapsule_GetPointer(arg, DEVICE_HANDLE_ID);

    #ifdef __APPLE__
    // Free the handle
    metal_release_device(&handle);
    #endif
}


// Initiates the KMeans class with the required params
static PyObject* kmeans_init(PyObject *self, PyObject *args) {

    int clusters;
    int iterations;
    int random_state;

    // GPU handle (optional)
    PyObject* device_capsule;

    if (!PyArg_ParseTuple(args, "iiiO", &clusters, &iterations, &random_state, &device_capsule)) {
        PyErr_SetString(PyExc_TypeError, "Unable to parse arguments");
        return NULL;
    }

    DeviceHandle handle = nullptr;

    // Ensure this is a Python capsule, if not let it be null
    if (PyCapsule_CheckExact(device_capsule)) {
        handle = PyCapsule_GetPointer(device_capsule, DEVICE_HANDLE_ID);
    }

    // Create the kmeans class reference
    KMeans *kmeans_handle = new KMeans((size_t)clusters, (size_t)iterations, random_state, 0.0001, handle);

    // Return it as a capsule
    return PyCapsule_New((void*)kmeans_handle, KMEANS_POINTER_ID, kmeans_destructor);
}


// Binding for fitting data to the model
static PyObject* kmeans_fit(PyObject *self, PyObject *args) {

    PyObject *handle;
    PyArrayObject *ndarray;

    // Parse the arguments
    if (!PyArg_ParseTuple(args, "OO", &handle, &ndarray)) {
        PyErr_SetString(PyExc_TypeError, "Unable to parse arguments");
        return NULL;
    }

    if (!PyCapsule_CheckExact(handle)) {
        PyErr_SetString(PyExc_TypeError, "First argument was not a capsule object");
        return NULL;
    }

    if (!PyArray_Check(ndarray)) {
        PyErr_SetString(PyExc_TypeError, "Second argument was not a numpy array");
        return NULL;
    }

    if (ndarray->nd != 2) {
        PyErr_SetString(PyExc_RuntimeError, "Data must be 2 dimensions");
        return NULL;
    }

    size_t rows = ndarray->dimensions[0];
    size_t cols = ndarray->dimensions[1];

    // Lightweight wrapper around the buffer
    Array<float> array((float*)ndarray->data, rows, cols, false);

    // Fit the data
    KMeans *kmeans = (KMeans*)PyCapsule_GetPointer(handle, KMEANS_POINTER_ID);
    kmeans->fit(array);

    return Py_None;
}


// Binding for predicting data
static PyObject* kmeans_predict(PyObject* self, PyObject *args) {

    PyObject *handle;
    PyArrayObject *ndarray;

    // Parse the arguments
    if (!PyArg_ParseTuple(args, "OO", &handle, &ndarray)) {
        PyErr_SetString(PyExc_TypeError, "Unable to parse arguments");
        return NULL;
    }

    if (!PyCapsule_CheckExact(handle)) {
        PyErr_SetString(PyExc_TypeError, "First argument was not a capsule object");
        return NULL;
    }

    if (!PyArray_Check(ndarray)) {
        PyErr_SetString(PyExc_TypeError, "Second argument was not a numpy array");
        return NULL;
    }

    if (ndarray->nd != 2) {
        PyErr_SetString(PyExc_RuntimeError, "Data must be 2 dimensions");
        return NULL;
    }

    size_t rows = ndarray->dimensions[0];
    size_t cols = ndarray->dimensions[1];

    // Lightweight wrapper around the buffer
    Array<float> array((float*)ndarray->data, rows, cols, false);

    KMeans *kmeans = (KMeans*)PyCapsule_GetPointer(handle, KMEANS_POINTER_ID);

    auto labels = kmeans->predict(array);

    long res_rows = (long)labels.rows();
    long res_cols = (long)labels.cols();

    npy_intp dims[2] = { res_rows, res_cols };

    int *labels_dst = new int[cols];
    memcpy(labels_dst, labels.data(), res_cols * sizeof(int));

    PyObject *ndarray_result = PyArray_SimpleNewFromData(2, dims, NPY_INT, labels_dst);
    PyArray_ENABLEFLAGS((PyArrayObject*)ndarray_result, NPY_ARRAY_OWNDATA);

    return ndarray_result;
}


static PyObject* kmeans_centroids(PyObject* self, PyObject *args) {

    // Handle to the kmeans instance
    PyObject *handle;

    if (!PyArg_ParseTuple(args, "O", &handle)) {
        PyErr_SetString(PyExc_TypeError, "Unable to parse arguments");
        return NULL;
    }

    if (!PyCapsule_CheckExact(handle)) {
        PyErr_SetString(PyExc_TypeError, "First argument was not a capsule object");
        return NULL;
    }

    KMeans *kmeans = (KMeans*)PyCapsule_GetPointer(handle, KMEANS_POINTER_ID);

    auto centroids = kmeans->centroids();

    long rows = (long)centroids.rows();
    long cols = (long)centroids.cols();

    npy_intp shape[2] = { rows, cols };

    size_t N = rows * cols;

    float *centroids_buffer = new float[N];
    memcpy(centroids_buffer, centroids.data(), N * sizeof(float));

    PyObject *ndarray_result = PyArray_SimpleNewFromData(2, shape, NPY_FLOAT, centroids_buffer);
    PyArray_ENABLEFLAGS((PyArrayObject*)ndarray_result, NPY_ARRAY_OWNDATA);

    return ndarray_result;
}


// Python binding to get information about the GPU if possible
static PyObject* device_retrieve_gpu(PyObject *self, PyObject *args) {

    gpu_device device;

    int result = -1;

    #ifdef __APPLE__
    result = metal_find_device(&device);
    #endif


    // Unable to get GPU so return nothing
    if (result != 0) {
        return Py_None;
    }

    PyObject *device_id = PyLong_FromLong((long)device.id);
    PyObject *device_name = PyUnicode_FromString(device.name);

    PyObject *device_attr = PyDict_New();
    PyDict_SetItemString(device_attr, "device_id", device_id);
    PyDict_SetItemString(device_attr, "device_name", device_name);

    return device_attr;
}


// Creates a GPU device handle
static PyObject* device_create_handle(PyObject *self, PyObject *args) {

    const char *lib_path;

    if (!PyArg_ParseTuple(args, "s", &lib_path)) {
        PyErr_SetString(PyExc_TypeError, "Unable to parse arguments");
        return NULL;
    }

    DeviceHandle handle = nullptr;

    int result = -1;
    
    #ifdef __APPLE__
    result = metal_init_device(&handle, lib_path);
    #endif

    if (result != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to initiate the GPU on this computer");
        return Py_None;
    }

    // Encapsulate the device handle pointer as a Python capsule to keep it alive
    PyObject *capsule = PyCapsule_New(handle, DEVICE_HANDLE_ID, device_destructor);
    
    // Return and save it
    return capsule;
}


// All methods to export to the main python package
static PyMethodDef export_methods[] = {

    // KMeans related bindings
    { "kmeans_init",    &kmeans_init,       METH_VARARGS, NULL },
    { "kmeans_fit",     &kmeans_fit,        METH_VARARGS, NULL },
    { "kmeans_predict", &kmeans_predict,    METH_VARARGS, NULL },
    { "kmeans_centroids", &kmeans_centroids,  METH_VARARGS, NULL },


    // GPU related bindings
    { "device_retrieve_gpu", &device_retrieve_gpu, METH_VARARGS, NULL },
    { "device_create_handle", &device_create_handle, METH_VARARGS, NULL },


    // TODO: Additional bindings here

    // Sentinel
    { NULL, NULL, 0, NULL },
};


static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "quickcluster._C",
    "Internal module for low level bindings",
    -1,
    export_methods
};


PyMODINIT_FUNC PyInit__C(void) {
    auto mod = PyModule_Create(&module);
    import_array();
    return mod;
}