/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue 01 Oct 2013 15:37:07 CEST
 *
 * @brief Pure python bindings for Blitz Arrays
 */

#include <blitz.array/helper.h>
#include <structmember.h>

#define ARRAY_MAXDIMS 4

typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  std::shared_ptr<void> bzarr;
  std::shared_ptr<PyObject> shape;
  std::shared_ptr<PyObject> dtype;

} PyBzArrayObject;

static char static_shape_str[] = "shape";
static char static_dtype_str[] = "dtype";
static char static_shape_doc[] = "a tuple indicating the shape of this array";
static char static_dtype_doc[] = "data type for every element in this array";

static PyMemberDef PyBzArray_members[] = {
    {
      static_shape_str,
      T_OBJECT_EX,
      offsetof(PyBzArrayObject, shape),
      READONLY,
      static_shape_doc
    },
    {
      static_dtype_str, 
      T_OBJECT_EX, 
      offsetof(PyBzArrayObject, dtype),
      READONLY,
      static_dtype_doc
    },
    {NULL}  /* Sentinel */
};

/**
 * Deallocates memory for an PyBzArrayObject object
 */
static void PyBzArray_dealloc(PyBzArrayObject* self) {

  //forces premature deallocation as free() will not respect C++ destruction
  self->bzarr.reset();
  self->shape.reset();
  self->dtype.reset();

  //calls free() on the PyObject itself
  self->ob_type->tp_free((PyObject*)self);
}

/**
 * Allocates memory and pre-initializes an PyBzArrayObject object
 */
static PyObject* PyBzArray_new(PyTypeObject* type, PyObject *args, 
    PyObject* kwds) {

  /* Allocates the python object itself */
  PyBzArrayObject* self = (PyBzArrayObject*)type->tp_alloc(type, 0);

  return reinterpret_cast<PyObject*>(self);
}

/** tests code for allocation
template<typename T, int N>
void delete_array(blitz::Array<T,N>* o) {
  std::cout << "deallocating array" << std::endl;
  delete o;
}
**/

/**
 * Converts any compatible sequence into a shape tuple
 */
static int PyBzArray_PySequence_AsShape(PyObject* obj, PyTupleObject** tuple) {

  if (!obj) {
    PyErr_SetString(PyExc_TypeError, "shape must not be NULL");
    return -1;
  }

  if (PyNumber_Check(obj)) {
    /* It is a number, user wants an array */
#if PY_VERSION_HEX >= 0x03000000
    auto intobj = pybz::detail::handle(PyNumber_Long(obj));
#else
    auto intobj = pybz::detail::handle(PyNumber_Int(obj));
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
  auto retval = pybz::detail::handle(PyTuple_New(ndim));
  for (Py_ssize_t i=0; i<ndim; ++i) {
    auto item = pybz::detail::handle(PySequence_GetItem(obj, i));
    if (!item) return 0;
    if (!PyNumber_Check(item.get())) {
      PyErr_Format(PyExc_RuntimeError, "element %" PY_FORMAT_SIZE_T "d from shape sequence should be an integer", i);
      return 0;
    }
#if PY_VERSION_HEX >= 0x03000000
    auto intobj = pybz::detail::handle(PyNumber_Long(item.get()));
#else
    auto intobj = pybz::detail::handle(PyNumber_Int(item.get()));
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
  (*tuple) = reinterpret_cast<PyTupleObject*>(pybz::detail::new_reference(retval));
  return 1;
}

static int PyBzArray_PyObject_AsDtype(PyObject* obj, PyArray_Descr** dtype) {

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
void PyBzArray_PyTuple_AsSsizeCArray(PyObject* o, Py_ssize_t* a) {
  Py_ssize_t ndim = PySequence_Fast_GET_SIZE(o);
  for (Py_ssize_t i=0; i<ndim; ++i) {
    a[i] = PyNumber_AsSsize_t(PySequence_Fast_GET_ITEM(o, i),
        PyExc_OverflowError);
  }
}

/**
 * Formal initialization of an Array object
 */
static int PyBzArray_init(PyBzArrayObject* self, PyObject *args,
    PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static char* kwlist[] = {static_shape_str, static_dtype_str, NULL};

  PyArray_Descr* dtype;
  PyTupleObject* shape;

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "O&O&", kwlist,
        &PyBzArray_PySequence_AsShape, &shape,
        &PyBzArray_PyObject_AsDtype, &dtype)
      )
    return -1;

  /* Creates auto-destructable containers for shape and dtype */
  std::shared_ptr<PyObject> ashape = pybz::detail::handle(reinterpret_cast<PyObject*>(shape));
  std::shared_ptr<PyObject> adtype = pybz::detail::handle(reinterpret_cast<PyObject*>(dtype));

  Py_ssize_t c_shape[ARRAY_MAXDIMS];
  PyBzArray_PyTuple_AsSsizeCArray(ashape.get(), c_shape);
  if (PyErr_Occurred()) return -1;

  /* at this point we have all data ready for the array creation */
  Py_ssize_t ndim = PySequence_Fast_GET_SIZE(ashape.get());
  try {
    self->bzarr = pybz::detail::allocate(dtype->type_num, ndim, c_shape);
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
static Py_ssize_t PyBzArray_len (PyBzArrayObject* self) {
  Py_ssize_t ndim = PySequence_Fast_GET_SIZE(self->shape.get());
  Py_ssize_t retval = 1;
  for (Py_ssize_t i=0; i<ndim; ++i) {
    retval *= PyNumber_AsSsize_t(PySequence_Fast_GET_ITEM(self->shape.get(), i), PyExc_OverflowError);
  }
  return retval;
}

static PyObject* PyBzArray_getitem(PyBzArrayObject* self,
    PyObject* item) {
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
    return pybz::detail::getitem(reinterpret_cast<PyArray_Descr*>(self->dtype.get()), ndim, self->bzarr, &k);

  }

  if (PySequence_Check(item)) {

    if (ndim != PySequence_Fast_GET_SIZE(item)) {
      PyErr_Format(PyExc_TypeError, "expected tuple of size %" PY_FORMAT_SIZE_T "d for accessing %" PY_FORMAT_SIZE_T "dD blitz::Array<>", ndim, ndim);
      return 0;
    }

    // if you get to this point, then the input tuple has the same size
    Py_ssize_t c_shape[ARRAY_MAXDIMS];
    PyBzArray_PyTuple_AsSsizeCArray(item, c_shape);
    if (PyErr_Occurred()) return 0;

    return pybz::detail::getitem(reinterpret_cast<PyArray_Descr*>(self->dtype.get()), ndim, self->bzarr, c_shape);

  }

  PyErr_SetString(PyExc_TypeError, "blitz::Array<> indexing requires a single integers (for 1D arrays) or sequences, for any rank size");
  return 0;
}

static int PyBzArray_setitem(PyBzArrayObject* self, PyObject* item,
    PyObject* value) {

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
    return pybz::detail::setitem(reinterpret_cast<PyArray_Descr*>(self->dtype.get()), ndim, self->bzarr, &k, value);

  }

  if (PySequence_Check(item)) {

    if (ndim != PySequence_Fast_GET_SIZE(item)) {
      PyErr_Format(PyExc_TypeError, "expected tuple of size %" PY_FORMAT_SIZE_T "d for accessing %" PY_FORMAT_SIZE_T "dD blitz::Array<>", ndim, ndim);
      return -1;
    }

    // if you get to this point, then the input tuple has the same size
    Py_ssize_t c_shape[ARRAY_MAXDIMS];
    PyBzArray_PyTuple_AsSsizeCArray(item, c_shape);
    if (PyErr_Occurred()) return -1;

    return pybz::detail::setitem(reinterpret_cast<PyArray_Descr*>(self->dtype.get()), ndim, self->bzarr, c_shape, value);

  }

  PyErr_SetString(PyExc_TypeError, "blitz::Array<> indexing requires a single integers (for 1D arrays) or sequences, for any rank size");
  return -1;
}

static PyMappingMethods PyBzArray_mapping = {
    (lenfunc)PyBzArray_len,
    (binaryfunc)PyBzArray_getitem,
    (objobjargproc)PyBzArray_setitem,
};

static PyObject* PyBzArray_ndarray_copy(PyBzArrayObject* self) {
  Py_ssize_t ndim = PySequence_Fast_GET_SIZE(self->shape.get());
  int typenum = (reinterpret_cast<PyArray_Descr*>(self->dtype.get()))->type_num;
  return pybz::detail::ndarray_copy(typenum, ndim, self->bzarr);
}

static PyObject* PyBzArray_ndarray_shallow(PyBzArrayObject* self) {
 
  Py_ssize_t ndim = PySequence_Fast_GET_SIZE(self->shape.get());
  int typenum = (reinterpret_cast<PyArray_Descr*>(self->dtype.get()))->type_num;
  
  PyObject* retval = pybz::detail::ndarray_shallow(typenum, ndim, self->bzarr);
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
static int PyBzArray_ndarray_shallow_able(PyBzArrayObject* self) {

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

static PyMethodDef PyBzArray_methods[] = {
    {
      static_as_ndarray_str,
      (PyCFunction)PyBzArray_ndarray_copy,
      METH_NOARGS,
      static_as_ndarray_doc
    },
    {
      static_as_shallow_ndarray_str,
      (PyCFunction)PyBzArray_ndarray_shallow,
      METH_NOARGS,
      static_as_shallow_ndarray_doc
    },
    {
      static_private_array_str,
      (PyCFunction)PyBzArray_ndarray_shallow,
      METH_NOARGS,
      static_private_array_doc
    },
    {NULL}  /* Sentinel */
};

static PyTypeObject PyBzArray_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                             /*ob_size*/
    "blitz.bzarray",               /*tp_name*/
    sizeof(PyBzArrayObject),       /*tp_basicsize*/
    0,                             /*tp_itemsize*/
    (destructor)PyBzArray_dealloc, /*tp_dealloc*/
    0,                             /*tp_print*/
    0,                             /*tp_getattr*/
    0,                             /*tp_setattr*/
    0,                             /*tp_compare*/
    0,                             /*tp_repr*/
    0,                             /*tp_as_number*/
    0,                             /*tp_as_sequence*/
    &PyBzArray_mapping,            /*tp_as_mapping*/
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
    PyBzArray_methods,             /* tp_methods */
    PyBzArray_members,             /* tp_members */
    0,                             /* tp_getset */
    0,                             /* tp_base */
    0,                             /* tp_dict */
    0,                             /* tp_descr_get */
    0,                             /* tp_descr_set */
    0,                             /* tp_dictoffset */
    (initproc)PyBzArray_init,      /* tp_init */
    0,                             /* tp_alloc */
    PyBzArray_new,                 /* tp_new */
};

static PyMethodDef array_methods[] = {
    {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC init_array(void)
{
  PyObject* m;

  /* makes sure we import numpy */
  pybz::detail::numpy_import_array();

  PyBzArray_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBzArray_Type) < 0) return;

  m = Py_InitModule3("_array", array_methods,
      "blitz::Array<> definition and generic functions");

  Py_INCREF(&PyBzArray_Type);
  PyModule_AddObject(m, "array", (PyObject *)&PyBzArray_Type);
}
