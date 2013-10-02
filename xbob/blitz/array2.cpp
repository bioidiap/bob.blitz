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

  /* Make sure shape is good */
  if (!PySequence_Check(obj)) {
    PyErr_SetString(PyExc_TypeError, "shape must be a sequence");
    return -1;
  }

  Py_ssize_t ndim = PySequence_Size(obj);
  if (ndim == 0 || ndim > ARRAY_MAXDIMS) {
    PyErr_Format(PyExc_TypeError, "shape must be a sequence with at least 1 and at most %d elements (you passed a sequence with %" PY_FORMAT_SIZE_T "d elements)", ARRAY_MAXDIMS, ndim);
    return -1;
  }

  /* Converts the input information into a shape */
  auto retval = bob::python::handle(PyTuple_New(ndim));
  for (Py_ssize_t i=0; i<ndim; ++i) {
    auto item = bob::python::handle(PySequence_GetItem(obj, i));
    if (!item) {
      PyErr_Format(PyExc_RuntimeError, "error obtaining element %" PY_FORMAT_SIZE_T "d from shape sequence", i);
      return -1;
    }
    if (!PyInt_Check(item.get()) && !PyLong_Check(item.get())) {
      PyErr_Format(PyExc_RuntimeError, "element %" PY_FORMAT_SIZE_T "d from shape sequence should be integral (int or long)", i);
      return -1;
    }
    PyTuple_SetItem(retval.get(), i, item.get());
  }

  /* At this point, you know everything went well */
  (*tuple) = reinterpret_cast<PyTupleObject*>(bob::python::new_reference(retval));
  return 0;
}

static int PyArray_DescrSupported(PyObject* obj, PyArray_Descr** dtype) {

  /* Make sure the dtype is good */
  if (!obj) {
    PyErr_SetString(PyExc_TypeError, "dtype must not be NULL");
    return -1;
  }
  
  if (!PyArray_DescrConverter2(obj, dtype)) return -1;

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
#endif
    default:
    {
      Py_DECREF(*dtype);
      PyErr_Format(PyExc_NotImplementedError, "no support for using type number %d to build blitz::Array<>", typenum);
      return -1;
    }
  }

  /* At this point, you know everything went well */
  return 0;
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

  Py_ssize_t ndim = PyTuple_Size(ashape.get());
  Py_ssize_t c_shape[ARRAY_MAXDIMS];
  for (Py_ssize_t i=0; i<ndim; ++i) {
    auto item = bob::python::handle(PySequence_GetItem(ashape.get(), i));
    c_shape[i] = PyLong_AsSsize_t(item.get());
    if (c_shape[i] == -1) {
      PyErr_Format(PyExc_RuntimeError, "error extracting a (signed) size from element %" PY_FORMAT_SIZE_T "d of input shape sequence", i);
      return -1;
    }
  }

  /* at this point we have all data ready for the array creation */
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

static char static_shape_doc[] = "a tuple indicating the shape of this array";
static char static_dtype_doc[] = "data type for every element in this array";
static PyMemberDef Array_members[] = {
    {static_shape_str, T_OBJECT_EX, offsetof(Array, shape), 0, static_shape_doc},
    {static_dtype_str, T_OBJECT_EX, offsetof(Array, dtype), 0, static_dtype_doc},
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
    0,                             /*tp_as_mapping*/
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
    0,                             /* tp_methods */
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

  array2_ArrayType.tp_new = PyType_GenericNew;
  if (PyType_Ready(&array2_ArrayType) < 0) return;

  m = Py_InitModule3("array2", array2_methods,
      "Blitz array definition and generic functions");

  Py_INCREF(&array2_ArrayType);
  PyModule_AddObject(m, "Array", (PyObject *)&array2_ArrayType);
}
