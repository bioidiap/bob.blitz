/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue 01 Oct 2013 15:37:07 CEST
 *
 * @brief Pure python bindings for Blitz Arrays
 */

#include "bob/py.h"
#include "structmember.h"

#define ARRAY_MAXDIMS 4

typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  std::shared_ptr<void> bzarr;
  std::shared_ptr<PyObject> shape;
  std::shared_ptr<PyObject> dtype;

} Array;

static char static_shape_str[] = "shape";
static char static_dtype_str[] = "dtype";
static char static_shape_doc[] = "a tuple indicating the shape of this array";
static char static_dtype_doc[] = "data type for every element in this array";

static PyMemberDef Array_members[] = {
    {static_shape_str, T_OBJECT_EX, offsetof(Array, shape), READONLY,
      static_shape_doc},
    {static_dtype_str, T_OBJECT_EX, offsetof(Array, dtype), READONLY,
      static_dtype_doc},
    {NULL}  /* Sentinel */
};

/**
 * Deallocates memory for an Array object
 */
static void Array_dealloc(Array* self) {
  self->ob_type->tp_free((PyObject*)self);
}

/**
 * Allocates memory and pre-initializes an Array object
 */
static PyObject* Array_new(PyTypeObject* type, PyObject *args, PyObject* kwds) {

  /* Allocates the python object itself */
  Array* self = (Array*)type->tp_alloc(type, 0);

  return reinterpret_cast<PyObject*>(self);
}

/**
 * Creates a new underlying Array object
 */
template<typename T, int N>
std::shared_ptr<void> allocate_array3(Py_ssize_t* shape) {
  blitz::TinyVector<int,N> tv_shape;
  for (int i=0; i<N; ++i) tv_shape(i) = shape[i];
  auto retval = std::make_shared<blitz::Array<T,N>>(tv_shape);
  return retval;
}

template<typename T>
std::shared_ptr<void> allocate_array2(Py_ssize_t ndim, Py_ssize_t* shape) {
  switch (ndim) {
    case 1: return allocate_array3<T,1>(shape);
    case 2: return allocate_array3<T,2>(shape);
    case 3: return allocate_array3<T,3>(shape);
    case 4: return allocate_array3<T,4>(shape);
    default:
      PyErr_Format(PyExc_TypeError, "cannot create blitz::Array<> array with %" PY_FORMAT_SIZE_T "d dimensions", ndim);
      return std::shared_ptr<void>();
  }
}

std::shared_ptr<void> allocate_array(int typenum, Py_ssize_t ndim, Py_ssize_t* shape) {
  switch (typenum) {
    case NPY_BOOL: return allocate_array2<bool>(ndim, shape);
    case NPY_INT8: return allocate_array2<int8_t>(ndim, shape);
    case NPY_INT16: return allocate_array2<int16_t>(ndim, shape);
    case NPY_INT32: return allocate_array2<int32_t>(ndim, shape);
    case NPY_INT64: return allocate_array2<int64_t>(ndim, shape);
    case NPY_UINT8: return allocate_array2<uint8_t>(ndim, shape);
    case NPY_UINT16: return allocate_array2<uint16_t>(ndim, shape);
    case NPY_UINT32: return allocate_array2<uint32_t>(ndim, shape);
    case NPY_UINT64: return allocate_array2<uint64_t>(ndim, shape);
    case NPY_FLOAT32: return allocate_array2<float>(ndim, shape);
    case NPY_FLOAT64: return allocate_array2<double>(ndim, shape);
#ifdef NPY_FLOAT128
    case NPY_FLOAT128: return allocate_array2<long double>(ndim, shape);
#endif
    case NPY_COMPLEX64: return allocate_array2<std::complex<float>>(ndim, shape);
    case NPY_COMPLEX128: return allocate_array2<std::complex<double>>(ndim, shape);
#ifdef NPY_COMPLEX256
    case NPY_COMPLEX256: return allocate_array2<std::complex<long double>>(ndim, shape);
#endif
    default:
      PyErr_Format(PyExc_TypeError, "cannot create array with data type number = %d", typenum);
      return std::shared_ptr<void>();
  }
}

/**
 * Converts any compatible sequence into a shape tuple
 */
static int PySequence_AsTuple(PyObject* obj, PyTupleObject** tuple) {

  if (!obj) {
    PyErr_SetString(PyExc_TypeError, "shape must not be NULL");
    return -1;
  }

  if (PyNumber_Check(obj)) {
    /* It is a number, user wants an array */
#if PY_VERSION_HEX >= 0x03000000
    auto intobj = bob::python::handle(PyNumber_Long(obj));
#else
    auto intobj = bob::python::handle(PyNumber_Int(obj));
#endif
    if (!intobj) return 0;
    Py_ssize_t k = PyNumber_AsSsize_t(intobj.get(), PyExc_OverflowError);
    if (k == -1 && PyErr_Occurred()) return 0;
    if (k <= 0) {
      PyErr_Format(PyExc_OverflowError, "error extracting a size from user input number (set to %" PY_FORMAT_SIZE_T "d) - shape elements should be > 0", k);
      return 0;
    }
    (*tuple) = reinterpret_cast<PyTupleObject*>(PyTuple_Pack(1, intobj.get()));
    return 1;
  }

  /* The other option is to have it as a sequence */
  if (!PySequence_Check(obj)) {
    PyErr_SetString(PyExc_TypeError, "shape must be a sequence of integers");
    return 0;
  }

  Py_ssize_t ndim = PySequence_Size(obj);
  if (ndim == 0 || ndim > ARRAY_MAXDIMS) {
    PyErr_Format(PyExc_TypeError, "shape must be a sequence with at least 1 and at most %d elements (you passed a sequence with %" PY_FORMAT_SIZE_T "d elements)", ARRAY_MAXDIMS, ndim);
    return 0;
  }

  /* Converts the input information into a shape */
  auto retval = bob::python::handle(PyTuple_New(ndim));
  for (Py_ssize_t i=0; i<ndim; ++i) {
    auto item = bob::python::handle(PySequence_GetItem(obj, i));
    if (!item) return 0;
    if (!PyNumber_Check(item.get())) {
      PyErr_Format(PyExc_RuntimeError, "element %" PY_FORMAT_SIZE_T "d from shape sequence should be an integer", i);
      return 0;
    }
#if PY_VERSION_HEX >= 0x03000000
    auto intobj = bob::python::handle(PyNumber_Long(item.get()));
#else
    auto intobj = bob::python::handle(PyNumber_Int(item.get()));
#endif
    if (!intobj) return 0;
    Py_ssize_t k = PyNumber_AsSsize_t(intobj.get(), PyExc_OverflowError);
    if (k == -1 && PyErr_Occurred()) return 0;
    if (k <= 0) {
      PyErr_Format(PyExc_OverflowError, "error extracting a size from element %" PY_FORMAT_SIZE_T "d (set to %" PY_FORMAT_SIZE_T "d) of input shape sequence - shape elements should be > 0", i, k);
      return 0;
    }
    PyTuple_SetItem(retval.get(), i, intobj.get());
  }

  /* At this point, you know everything went well */
  (*tuple) = reinterpret_cast<PyTupleObject*>(bob::python::new_reference(retval));
  return 1;
}

static int PyArray_DescrSupported(PyObject* obj, PyArray_Descr** dtype) {

  /* Make sure the dtype is good */
  if (!obj) {
    PyErr_SetString(PyExc_TypeError, "dtype must not be NULL");
    return 0;
  }

  if (!PyArray_DescrConverter2(obj, dtype)) return 0;

  int typenum = (*dtype)->type_num;

  switch (typenum) {
    case NPY_BOOL:
    case NPY_UINT8:
    case NPY_UINT16:
    case NPY_UINT32:
    case NPY_UINT64:
    case NPY_INT8:
    case NPY_INT16:
    case NPY_INT32:
    case NPY_INT64:
    case NPY_FLOAT32:
    case NPY_FLOAT64:
#ifdef NPY_FLOAT128
    case NPY_FLOAT128:
#endif
    case NPY_COMPLEX64:
    case NPY_COMPLEX128:
#ifdef NPY_COMPLEX256
    case NPY_COMPLEX256:
      break;
#endif
    default:
    {
      Py_DECREF(*dtype);
      PyErr_Format(PyExc_NotImplementedError, "no support for using type number %d to build blitz::Array<>", typenum);
      return 0;
    }
  }

  /* At this point, you know everything went well */
  return 1;
}

/**
 * Converts a shape information into a C array of Py_ssize_t
 */
void Fast_PyTuple_AsSsizeArray(PyObject* o, Py_ssize_t* a) {
  Py_ssize_t ndim = PySequence_Fast_GET_SIZE(o);
  for (Py_ssize_t i=0; i<ndim; ++i) {
    a[i] = PyNumber_AsSsize_t(PySequence_Fast_GET_ITEM(o, i),
        PyExc_OverflowError);
  }
}

/**
 * Formal initialization of an Array object
 */
static int Array_init(Array* self, PyObject *args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static char* kwlist[] = {static_shape_str, static_dtype_str, NULL};

  PyArray_Descr* dtype;
  PyTupleObject* shape;

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "O&O&", kwlist,
        &PySequence_AsTuple, &shape,
        &PyArray_DescrSupported, &dtype)
      )
    return -1;

  /* Creates auto-destructable containers for shape and dtype */
  std::shared_ptr<PyObject> ashape = bob::python::handle(reinterpret_cast<PyObject*>(shape));
  std::shared_ptr<PyObject> adtype = bob::python::handle(reinterpret_cast<PyObject*>(dtype));

  Py_ssize_t c_shape[ARRAY_MAXDIMS];
  Fast_PyTuple_AsSsizeArray(ashape.get(), c_shape);
  if (PyErr_Occurred()) return -1;

  /* at this point we have all data ready for the array creation */
  Py_ssize_t ndim = PySequence_Fast_GET_SIZE(ashape.get());
  try {
    self->bzarr = allocate_array(dtype->type_num, ndim, c_shape);
    if (!self->bzarr) return -1;
  }
  catch (std::exception& e) {
    PyErr_Format(PyExc_RuntimeError, "std::exception caught when creating array with %" PY_FORMAT_SIZE_T "d dimensions and type number = %d: %s", ndim, dtype->type_num, e.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "unknown exception caught when creating array with %" PY_FORMAT_SIZE_T "d dimensions and type number = %d", ndim, dtype->type_num);
    return -1;
  }

  /* update internal variables */
  self->dtype = adtype;
  self->shape = ashape;

  return 0; ///< SUCCESS
}

/**
 * Methods for Sequence operation
 */
static Py_ssize_t Array_len (Array* self) {
  Py_ssize_t ndim = PySequence_Fast_GET_SIZE(self->shape.get());
  Py_ssize_t retval = 1;
  for (Py_ssize_t i=0; i<ndim; ++i) {
    retval *= PyNumber_AsSsize_t(PySequence_Fast_GET_ITEM(self->shape.get(), i), PyExc_OverflowError);
  }
  return retval;
}

/**
 * Returns a given item from the blitz::Array<>
 */
template <typename T>
PyObject* getitem_array2(PyArray_Descr* dtype, Py_ssize_t ndim, std::shared_ptr<void> bz, Py_ssize_t* pos) {
  switch (ndim) {
    case 1:
      {
        T& val = (*reinterpret_cast<blitz::Array<T,1>*>(bz.get()))((int)pos[0]);
        return PyArray_Scalar(&val, dtype, 0);
      }
    case 2:
      {
        T& val = (*reinterpret_cast<blitz::Array<T,2>*>(bz.get()))((int)pos[0], (int)pos[1]);
        return PyArray_Scalar(&val, dtype, 0);
      }
    case 3:
      {
        T& val = (*reinterpret_cast<blitz::Array<T,2>*>(bz.get()))((int)pos[0], (int)pos[1], (int)pos[2]);
        return PyArray_Scalar(&val, dtype, 0);
      }
    case 4:
      {
        T& val = (*reinterpret_cast<blitz::Array<T,2>*>(bz.get()))((int)pos[0], (int)pos[1], (int)pos[2], (int)pos[3]);
        return PyArray_Scalar(&val, dtype, 0);
      }
    default:
      PyErr_Format(PyExc_TypeError, "cannot index blitz::Array<> array with %" PY_FORMAT_SIZE_T "d dimensions", ndim);
      return 0;
  }
}

PyObject* getitem_array(PyArray_Descr* dtype, Py_ssize_t ndim, std::shared_ptr<void> bz, Py_ssize_t* pos) {
  switch (dtype->type_num) {
    case NPY_BOOL: return getitem_array2<bool>(dtype, ndim, bz, pos);
    case NPY_INT8: return getitem_array2<int8_t>(dtype, ndim, bz, pos);
    case NPY_INT16: return getitem_array2<int16_t>(dtype, ndim, bz, pos);
    case NPY_INT32: return getitem_array2<int32_t>(dtype, ndim, bz, pos);
    case NPY_INT64: return getitem_array2<int64_t>(dtype, ndim, bz, pos);
    case NPY_UINT8: return getitem_array2<uint8_t>(dtype, ndim, bz, pos);
    case NPY_UINT16: return getitem_array2<uint16_t>(dtype, ndim, bz, pos);
    case NPY_UINT32: return getitem_array2<uint32_t>(dtype, ndim, bz, pos);
    case NPY_UINT64: return getitem_array2<uint64_t>(dtype, ndim, bz, pos);
    case NPY_FLOAT32: return getitem_array2<float>(dtype, ndim, bz, pos);
    case NPY_FLOAT64: return getitem_array2<double>(dtype, ndim, bz, pos);
#ifdef NPY_FLOAT128
    case NPY_FLOAT128: return getitem_array2<long double>(dtype, ndim, bz, pos);
#endif
    case NPY_COMPLEX64: return getitem_array2<std::complex<float>>(dtype, ndim, bz, pos);
    case NPY_COMPLEX128: return getitem_array2<std::complex<double>>(dtype, ndim, bz, pos);
#ifdef NPY_COMPLEX256
    case NPY_COMPLEX256: return getitem_array2<std::complex<long double>>(dtype, ndim, bz, pos);
#endif
    default:
      PyErr_Format(PyExc_TypeError, "cannot index array with data type number = %d", dtype->type_num);
      return 0;
  }
}

static PyObject* Array_getitem(Array* self, PyObject* item) {
  Py_ssize_t ndim = PySequence_Fast_GET_SIZE(self->shape.get());

  if (PyNumber_Check(item)) {

    if (ndim != 1) {
      PyErr_Format(PyExc_TypeError, "expected number for accessing 1D blitz::Array<>");
      return 0;
    }

    // if you get to this point, the user has passed single number
    Py_ssize_t k = PyNumber_AsSsize_t(item, PyExc_IndexError);
    if (k == -1 && PyErr_Occurred()) return 0;
    Py_ssize_t extent = PyNumber_AsSsize_t(PySequence_Fast_GET_ITEM(self->shape.get(), 0), PyExc_OverflowError);
    if (k < 0) k += extent;
    if (k < 0 || k >= extent) {
      PyErr_SetString(PyExc_IndexError, "array index out of range");
      return 0;
    }

    // if you survived to this point, the value `k' is within range
    return getitem_array(reinterpret_cast<PyArray_Descr*>(self->dtype.get()), ndim, self->bzarr, &k);

  }

  if (PySequence_Check(item)) {

    if (ndim != PySequence_Fast_GET_SIZE(item)) {
      PyErr_Format(PyExc_TypeError, "expected tuple of size %" PY_FORMAT_SIZE_T "d for accessing %" PY_FORMAT_SIZE_T "dD blitz::Array<>", ndim, ndim);
      return 0;
    }

    // if you get to this point, then the input tuple has the same size
    Py_ssize_t c_shape[ARRAY_MAXDIMS];
    Fast_PyTuple_AsSsizeArray(item, c_shape);
    if (PyErr_Occurred()) return 0;

    return getitem_array(reinterpret_cast<PyArray_Descr*>(self->dtype.get()), ndim, self->bzarr, c_shape);

  }

  PyErr_SetString(PyExc_TypeError, "blitz::Array<> indexing requires a single integers (for 1D arrays) or sequences, for any rank size");
  return 0;
}

/**
 * Sets a given item from the blitz::Array<>
 */
template <typename T>
int setitem_array2(Py_ssize_t ndim, std::shared_ptr<void> bz, Py_ssize_t* pos, PyObject* value) {
  switch (ndim) {
    case 1:
      {
        T tmp = bob::python::extract<T>(value);
        if (PyErr_Occurred()) return -1;
        (*reinterpret_cast<blitz::Array<T,1>*>(bz.get()))((int)pos[0]) = tmp;
        return 0;
      }
    case 2:
      {
        T tmp = bob::python::extract<T>(value);
        if (PyErr_Occurred()) return -1;
        (*reinterpret_cast<blitz::Array<T,2>*>(bz.get()))((int)pos[0], (int)pos[1]) = tmp;
        return 0;
      }
    case 3:
      {
        T tmp = bob::python::extract<T>(value);
        if (PyErr_Occurred()) return -1;
        (*reinterpret_cast<blitz::Array<T,3>*>(bz.get()))((int)pos[0], (int)pos[1], (int)pos[2]) = tmp;
        return 0;
      }
    case 4:
      {
        T tmp = bob::python::extract<T>(value);
        if (PyErr_Occurred()) return -1;
        (*reinterpret_cast<blitz::Array<T,4>*>(bz.get()))((int)pos[0], (int)pos[1], (int)pos[2], (int)pos[3]) = tmp;
        return 0;
      }
    default:
      PyErr_Format(PyExc_TypeError, "cannot index blitz::Array<> array with %" PY_FORMAT_SIZE_T "d dimensions", ndim);
      return -1;
  }
}

int setitem_array(PyArray_Descr* dtype, Py_ssize_t ndim, std::shared_ptr<void> bz, Py_ssize_t* pos, PyObject* value) {
  switch (dtype->type_num) {
    case NPY_BOOL: return setitem_array2<bool>(ndim, bz, pos, value);
    case NPY_INT8: return setitem_array2<int8_t>(ndim, bz, pos, value);
    case NPY_INT16: return setitem_array2<int16_t>(ndim, bz, pos, value);
    case NPY_INT32: return setitem_array2<int32_t>(ndim, bz, pos, value);
    case NPY_INT64: return setitem_array2<int64_t>(ndim, bz, pos, value);
    case NPY_UINT8: return setitem_array2<uint8_t>(ndim, bz, pos, value);
    case NPY_UINT16: return setitem_array2<uint16_t>(ndim, bz, pos, value);
    case NPY_UINT32: return setitem_array2<uint32_t>(ndim, bz, pos, value);
    case NPY_UINT64: return setitem_array2<uint64_t>(ndim, bz, pos, value);
    case NPY_FLOAT32: return setitem_array2<float>(ndim, bz, pos, value);
    case NPY_FLOAT64: return setitem_array2<double>(ndim, bz, pos, value);
#ifdef NPY_FLOAT128
    case NPY_FLOAT128: return setitem_array2<long double>(ndim, bz, pos, value);
#endif
    case NPY_COMPLEX64: return setitem_array2<std::complex<float>>(ndim, bz, pos, value);
    case NPY_COMPLEX128: return setitem_array2<std::complex<double>>(ndim, bz, pos, value);
#ifdef NPY_COMPLEX256
    case NPY_COMPLEX256: return setitem_array2<std::complex<long double>>(ndim, bz, pos, value);
#endif
    default:
      PyErr_Format(PyExc_TypeError, "cannot index array with data type number = %d", dtype->type_num);
      return -1;
  }
}

static int Array_setitem(Array* self, PyObject* item, PyObject* value) {
  Py_ssize_t ndim = PySequence_Fast_GET_SIZE(self->shape.get());

  if (PyNumber_Check(item)) {

    if (ndim != 1) {
      PyErr_Format(PyExc_TypeError, "expected number for accessing 1D blitz::Array<>");
      return -1;
    }

    // if you get to this point, the user has passed single number
    Py_ssize_t k = PyNumber_AsSsize_t(item, PyExc_IndexError);
    if (k == -1 && PyErr_Occurred()) return -1;
    Py_ssize_t extent = PyNumber_AsSsize_t(PySequence_Fast_GET_ITEM(self->shape.get(), 0), PyExc_OverflowError);
    if (k < 0) k += extent;
    if (k < 0 || k >= extent) {
      PyErr_SetString(PyExc_IndexError, "array index out of range");
      return -1;
    }

    // if you survived to this point, the value `k' is within range
    return setitem_array(reinterpret_cast<PyArray_Descr*>(self->dtype.get()), ndim, self->bzarr, &k, value);

  }

  if (PySequence_Check(item)) {

    if (ndim != PySequence_Fast_GET_SIZE(item)) {
      PyErr_Format(PyExc_TypeError, "expected tuple of size %" PY_FORMAT_SIZE_T "d for accessing %" PY_FORMAT_SIZE_T "dD blitz::Array<>", ndim, ndim);
      return -1;
    }

    // if you get to this point, then the input tuple has the same size
    Py_ssize_t c_shape[ARRAY_MAXDIMS];
    Fast_PyTuple_AsSsizeArray(item, c_shape);
    if (PyErr_Occurred()) return -1;

    return setitem_array(reinterpret_cast<PyArray_Descr*>(self->dtype.get()), ndim, self->bzarr, c_shape, value);

  }

  PyErr_SetString(PyExc_TypeError, "blitz::Array<> indexing requires a single integers (for 1D arrays) or sequences, for any rank size");
  return -1;
}

static PyMappingMethods Array_mapping = {
    (lenfunc)Array_len, /* sq_length */
    (binaryfunc)Array_getitem,
    (objobjargproc)Array_setitem,
};

template <typename T>
PyObject* ndarray_copy_array2(Py_ssize_t ndim, std::shared_ptr<void> bz) {
  switch (ndim) {
    case 1:
      {
        return bob::python::ndarray_copy(*reinterpret_cast<blitz::Array<T,1>*>(bz.get()));
      }
    case 2:
      {
        return bob::python::ndarray_copy(*reinterpret_cast<blitz::Array<T,2>*>(bz.get()));
      }
    case 3:
      {
        return bob::python::ndarray_copy(*reinterpret_cast<blitz::Array<T,3>*>(bz.get()));
      }
    case 4:
      {
        return bob::python::ndarray_copy(*reinterpret_cast<blitz::Array<T,4>*>(bz.get()));
      }
    default:
      PyErr_Format(PyExc_TypeError, "cannot convert blitz::Array<> to numpy.ndarray with number of dimensions = %" PY_FORMAT_SIZE_T "d", ndim);
      return 0;
  }
}

PyObject* ndarray_copy_array(int typenum, Py_ssize_t ndim, std::shared_ptr<void> bz) {
  switch (typenum) {
    case NPY_BOOL: return ndarray_copy_array2<bool>(ndim, bz);
    case NPY_INT8: return ndarray_copy_array2<int8_t>(ndim, bz);
    case NPY_INT16: return ndarray_copy_array2<int16_t>(ndim, bz);
    case NPY_INT32: return ndarray_copy_array2<int32_t>(ndim, bz);
    case NPY_INT64: return ndarray_copy_array2<int64_t>(ndim, bz);
    case NPY_UINT8: return ndarray_copy_array2<uint8_t>(ndim, bz);
    case NPY_UINT16: return ndarray_copy_array2<uint16_t>(ndim, bz);
    case NPY_UINT32: return ndarray_copy_array2<uint32_t>(ndim, bz);
    case NPY_UINT64: return ndarray_copy_array2<uint64_t>(ndim, bz);
    case NPY_FLOAT32: return ndarray_copy_array2<float>(ndim, bz);
    case NPY_FLOAT64: return ndarray_copy_array2<double>(ndim, bz);
#ifdef NPY_FLOAT128
    case NPY_FLOAT128: return ndarray_copy_array2<long double>(ndim, bz);
#endif
    case NPY_COMPLEX64: return ndarray_copy_array2<std::complex<float>>(ndim, bz);
    case NPY_COMPLEX128: return ndarray_copy_array2<std::complex<double>>(ndim, bz);
#ifdef NPY_COMPLEX256
    case NPY_COMPLEX256: return ndarray_copy_array2<std::complex<long double>>(ndim, bz);
#endif
    default:
      PyErr_Format(PyExc_TypeError, "cannot convert blitz::Array<> to numpy.ndarray with data type number = %d", typenum);
      return 0;
  }
}

static PyObject* Array_ndarray_copy(Array* self) {
  Py_ssize_t ndim = PySequence_Fast_GET_SIZE(self->shape.get());
  int typenum = (reinterpret_cast<PyArray_Descr*>(self->dtype.get()))->type_num;
  return ndarray_copy_array(typenum, ndim, self->bzarr);
}

template <typename T>
PyObject* ndarray_shallow_array2(Py_ssize_t ndim, std::shared_ptr<void> bz) {
  switch (ndim) {
    case 1:
      {
        return bob::python::ndarray_shallow(*reinterpret_cast<blitz::Array<T,1>*>(bz.get()));
      }
    case 2:
      {
        return bob::python::ndarray_shallow(*reinterpret_cast<blitz::Array<T,2>*>(bz.get()));
      }
    case 3:
      {
        return bob::python::ndarray_shallow(*reinterpret_cast<blitz::Array<T,3>*>(bz.get()));
      }
    case 4:
      {
        return bob::python::ndarray_shallow(*reinterpret_cast<blitz::Array<T,4>*>(bz.get()));
      }
    default:
      PyErr_Format(PyExc_TypeError, "cannot convert blitz::Array<> to numpy.ndarray with number of dimensions = %" PY_FORMAT_SIZE_T "d", ndim);
      return 0;
  }
}

PyObject* ndarray_shallow_array(int typenum, Py_ssize_t ndim, std::shared_ptr<void> bz) {
  switch (typenum) {
    case NPY_BOOL: return ndarray_shallow_array2<bool>(ndim, bz);
    case NPY_INT8: return ndarray_shallow_array2<int8_t>(ndim, bz);
    case NPY_INT16: return ndarray_shallow_array2<int16_t>(ndim, bz);
    case NPY_INT32: return ndarray_shallow_array2<int32_t>(ndim, bz);
    case NPY_INT64: return ndarray_shallow_array2<int64_t>(ndim, bz);
    case NPY_UINT8: return ndarray_shallow_array2<uint8_t>(ndim, bz);
    case NPY_UINT16: return ndarray_shallow_array2<uint16_t>(ndim, bz);
    case NPY_UINT32: return ndarray_shallow_array2<uint32_t>(ndim, bz);
    case NPY_UINT64: return ndarray_shallow_array2<uint64_t>(ndim, bz);
    case NPY_FLOAT32: return ndarray_shallow_array2<float>(ndim, bz);
    case NPY_FLOAT64: return ndarray_shallow_array2<double>(ndim, bz);
#ifdef NPY_FLOAT128
    case NPY_FLOAT128: return ndarray_shallow_array2<long double>(ndim, bz);
#endif
    case NPY_COMPLEX64: return ndarray_shallow_array2<std::complex<float>>(ndim, bz);
    case NPY_COMPLEX128: return ndarray_shallow_array2<std::complex<double>>(ndim, bz);
#ifdef NPY_COMPLEX256
    case NPY_COMPLEX256: return ndarray_shallow_array2<std::complex<long double>>(ndim, bz);
#endif
    default:
      PyErr_Format(PyExc_TypeError, "cannot convert blitz::Array<> to numpy.ndarray with data type number = %d", typenum);
      return 0;
  }
}

static PyObject* Array_ndarray_shallow(Array* self) {
 
  Py_ssize_t ndim = PySequence_Fast_GET_SIZE(self->shape.get());
  int typenum = (reinterpret_cast<PyArray_Descr*>(self->dtype.get()))->type_num;
  
  PyObject* retval = ndarray_shallow_array(typenum, ndim, self->bzarr);
  if (!retval) return 0;

  // sets numpy.ndarray base for the new array
#if NPY_FEATURE_VERSION < NUMPY17_API /* NumPy C-API version >= 1.7 */
  PyArray_BASE(reinterpret_cast<PyArrayObject*>(retval)) =
    reinterpret_cast<PyObject*>(self);
#else
  if (PyArray_SetBaseObject(reinterpret_cast<PyArrayObject*>(retval),
      reinterpret_cast<PyObject*>(self)) != 0) {
    Py_DECREF(retval);
    return 0;
  }
#endif
  Py_INCREF(reinterpret_cast<PyObject*>(self));

  return retval;

}

/**
static int Array_ndarray_shallow_able(Array* self) {

  // TODO: checks that the array is well-behaved
  if(!a.isStorageContiguous()) {
    PyErr_SetString(PyExc_RuntimeError, "blitz::Array<> is not C-contiguous and cannot be mapped into a read-only numpy.ndarray");
    return 0;
  }

  for(int i=0; i<a.rank(); ++i) {
    if(!(a.isRankStoredAscending(i) && a.ordering(i)==a.rank()-1-i)) {
      PyErr_Format(PyExc_RuntimeError, "dimension %d of blitz::Array<> is not stored in ascending order and cannot be mapped into a read-only numpy.ndarray", i);
      return 0;
    }
  }
}
**/

static char static_private_array_str[] = "__array__";
static char static_private_array_doc[] = "numpy.ndarray accessor (choses the fastest possible conversion path)";
static char static_as_ndarray_str[] = "as_ndarray";
static char static_as_ndarray_doc[] = "returns a copy of the blitz::Array<> as a numpy.ndarray";
static char static_as_shallow_ndarray_str[] = "as_shallow_ndarray";
static char static_as_shallow_ndarray_doc[] = "returns a (read-only) shallow copy of the blitz::Array<> as a numpy.ndarray";

static PyMethodDef Array_methods[] = {
    {
      static_as_ndarray_str,
      (PyCFunction)Array_ndarray_copy,
      METH_NOARGS,
      static_as_ndarray_doc
    },
    {
      static_as_shallow_ndarray_str,
      (PyCFunction)Array_ndarray_shallow,
      METH_NOARGS,
      static_as_shallow_ndarray_doc
    },
    {
      static_private_array_str,
      (PyCFunction)Array_ndarray_shallow,
      METH_NOARGS,
      static_private_array_doc
    },
    {NULL}  /* Sentinel */
};

static PyTypeObject array2_ArrayType = {
    PyObject_HEAD_INIT(NULL)
    0,                             /*ob_size*/
    "array2.Array",                /*tp_name*/
    sizeof(Array),                 /*tp_basicsize*/
    0,                             /*tp_itemsize*/
    (destructor)Array_dealloc,     /*tp_dealloc*/
    0,                             /*tp_print*/
    0,                             /*tp_getattr*/
    0,                             /*tp_setattr*/
    0,                             /*tp_compare*/
    0,                             /*tp_repr*/
    0,                             /*tp_as_number*/
    0,                             /*tp_as_sequence*/
    &Array_mapping,                /*tp_as_mapping*/
    0,                             /*tp_hash */
    0,                             /*tp_call*/
    0,                             /*tp_str*/
    0,                             /*tp_getattro*/
    0,                             /*tp_setattro*/
    0,                             /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /*tp_flags*/
    "An N-dimensional blitz::Array<> pythonic representation", /* tp_doc */
    0,		                         /* tp_traverse */
    0,		                         /* tp_clear */
    0,		                         /* tp_richcompare */
    0,		                         /* tp_weaklistoffset */
    0,		                         /* tp_iter */
    0,		                         /* tp_iternext */
    Array_methods,                 /* tp_methods */
    Array_members,                 /* tp_members */
    0,                             /* tp_getset */
    0,                             /* tp_base */
    0,                             /* tp_dict */
    0,                             /* tp_descr_get */
    0,                             /* tp_descr_set */
    0,                             /* tp_dictoffset */
    (initproc)Array_init,          /* tp_init */
    0,                             /* tp_alloc */
    Array_new,                     /* tp_new */
};

static PyMethodDef array2_methods[] = {
    {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC initarray2(void)
{
  PyObject* m;

  /* makes sure we import numpy */
  bob::python::bob_import_array();

  array2_ArrayType.tp_new = PyType_GenericNew;
  if (PyType_Ready(&array2_ArrayType) < 0) return;

  m = Py_InitModule3("array2", array2_methods,
      "Blitz array definition and generic functions");

  Py_INCREF(&array2_ArrayType);
  PyModule_AddObject(m, "Array", (PyObject *)&array2_ArrayType);
}
