/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue 01 Oct 2013 15:37:07 CEST
 *
 * @brief Pure python bindings for Blitz Arrays
 */

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif
#define BLITZ_ARRAY_MODULE
#include <blitz.array/capi.h>

#include <structmember.h>

static char static_shape_str[] = "shape";
static char static_dtype_str[] = "dtype";

/**
 * Formal initialization of an Array object
 */
static int PyBlitzArray_init(PyBlitzArrayObject* self, PyObject *args,
    PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static char* kwlist[] = {static_shape_str, static_dtype_str, NULL};

  PyBlitzArrayObject shape;
  PyBlitzArrayObject* shape_p = &shape;
  int type_num = -1;
  int* type_num_p = &type_num;

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "O&O&", kwlist,
        &PyBlitzArray_IndexConverter, &shape_p,
        &PyBlitzArray_TypenumConverter, &type_num_p)
      )
    return -1; ///< FAILURE

  /* Checks if none of the shape positions are zero */
  for (Py_ssize_t i=0; i<shape.ndim; ++i) {
    if (shape.shape[i] == 0) {
      PyErr_Format(PyExc_ValueError, "shape values should not be 0, but one was found at position %" PY_FORMAT_SIZE_T "d of input sequence", i);
      return -1; ///< FAILURE
    }
  }

  PyBlitzArrayObject* tmp = reinterpret_cast<PyBlitzArrayObject*>(PyBlitzArray_SimpleNew(type_num, shape.ndim, shape.shape));
  if (!tmp) return -1;

  /* Copies the new object to the pre-allocated one */
  (*self) = (*tmp);
  tmp->ob_type->tp_free((PyObject*)tmp); ///< Deallocates only the Python stuff

  return 0; ///< SUCCESS
}

/**
 * Methods for Sequence operation
 */
static Py_ssize_t PyBlitzArray_len (PyBlitzArrayObject* self) {
  Py_ssize_t retval = 1;
  for (Py_ssize_t i=0; i<self->ndim; ++i) retval *= self->shape[i];
  return retval;
}

static PyObject* PyBlitzArray_getitem(PyBlitzArrayObject* self,
    PyObject* item) {

  if (PyNumber_Check(item)) {

    if (self->ndim != 1) {
      PyErr_Format(PyExc_TypeError, "expected number for accessing 1D blitz::Array<>");
      return 0;
    }

    // if you get to this point, the user has passed single number
    Py_ssize_t k = PyNumber_AsSsize_t(item, PyExc_IndexError);
    return PyBlitzArray_GetItem(self, &k);

  }

  if (PySequence_Check(item)) {

    if (self->ndim != PySequence_Fast_GET_SIZE(item)) {
      PyErr_Format(PyExc_TypeError, "expected tuple of size %" PY_FORMAT_SIZE_T "d for accessing %" PY_FORMAT_SIZE_T "dD blitz::Array<>", self->ndim, self->ndim);
      return 0;
    }

    // if you get to this point, then the input tuple has the same size
    PyBlitzArrayObject shape;
    PyBlitzArrayObject* shape_p = &shape;
    if (!PyBlitzArray_IndexConverter(item, &shape_p)) return 0;
    return PyBlitzArray_GetItem(self, shape.shape);

  }

  PyErr_Format(PyExc_TypeError, "blitz::Array<%s,%" PY_FORMAT_SIZE_T "d> indexing requires a single integers (for 1D arrays) or sequences, for any rank size", PyBlitzArray_TypenumAsString(self->type_num), self->ndim);
  return 0;
}

static int PyBlitzArray_setitem(PyBlitzArrayObject* self, PyObject* item,
    PyObject* value) {

  if (PyNumber_Check(item)) {

    if (self->ndim != 1) {
      PyErr_Format(PyExc_TypeError, "expected sequence for accessing blitz::Array<%s,%" PY_FORMAT_SIZE_T "d>", PyBlitzArray_TypenumAsString(self->type_num), self->ndim);
      return -1;
    }

    // if you get to this point, the user has passed single number
    Py_ssize_t k = PyNumber_AsSsize_t(item, PyExc_IndexError);
    return PyBlitzArray_SetItem(self, &k, value);

  }

  if (PySequence_Check(item)) {

    if (self->ndim != PySequence_Fast_GET_SIZE(item)) {
      PyErr_Format(PyExc_TypeError, "expected sequence of size %" PY_FORMAT_SIZE_T "d for accessing blitz::Array<%s,%" PY_FORMAT_SIZE_T "d>", self->ndim, PyBlitzArray_TypenumAsString(self->type_num), self->ndim);
      return -1;
    }

    // if you get to this point, then the input tuple has the same size
    PyBlitzArrayObject shape;
    PyBlitzArrayObject* shape_p = &shape;
    if (!PyBlitzArray_IndexConverter(item, &shape_p)) return 0;
    return PyBlitzArray_SetItem(self, shape.shape, value);

  }

  PyErr_Format(PyExc_TypeError, "blitz::Array<%s,%" PY_FORMAT_SIZE_T "d> assignment requires a single integers (for 1D arrays) or sequences, for any rank size", PyBlitzArray_TypenumAsString(self->type_num), self->ndim);
  return -1;
}

static PyMappingMethods PyBlitzArray_mapping = {
    (lenfunc)PyBlitzArray_len,
    (binaryfunc)PyBlitzArray_getitem,
    (objobjargproc)PyBlitzArray_setitem,
};

/**
static int PyBlitzArray_ndarray_shallow_able(PyBlitzArrayObject* self) {

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
static char static_as_shallow_ndarray_doc[] = "returns a shallow copy of the blitz::Array<> as a numpy.ndarray";

static PyMethodDef PyBlitzArray_methods[] = {
    {
      static_as_ndarray_str,
      (PyCFunction)PyBlitzArray_AsNumpyNDArrayCopy,
      METH_NOARGS,
      static_as_ndarray_doc
    },
    {
      static_as_shallow_ndarray_str,
      (PyCFunction)PyBlitzArray_AsShallowNumpyNDArray,
      METH_NOARGS,
      static_as_shallow_ndarray_doc
    },
    {
      static_private_array_str,
      (PyCFunction)PyBlitzArray_AsAnyNumpyNDArray,
      METH_NOARGS,
      static_private_array_doc
    },
    {NULL}  /* Sentinel */
};

/* Property API */
static char static_shape_doc[] = "a tuple indicating the shape of this array";
static char static_dtype_doc[] = "the numpy.dtype equivalent for every element in this array";
static PyGetSetDef PyBlitzArray_getseters[] = {
    {
      static_dtype_str, 
      (getter)PyBlitzArray_DTYPE,
      0,
      static_dtype_doc,
      0,
    },
    {
      static_shape_str,
      (getter)PyBlitzArray_PYSHAPE,
      0,
      static_shape_doc,
      0,
    },
    {NULL}  /* Sentinel */
};

/* Stringification */
static PyObject* PyBlitzArray_str(PyBlitzArrayObject* o) {
  PyObject* nd = PyBlitzArray_AsAnyNumpyNDArray(o);
  if (!nd) {
    PyErr_Print();
    PyErr_SetString(PyExc_RuntimeError, "could not convert blitz::Array<> into numpy ndarray for str() method call");
    return 0;
  }
  PyObject* retval = PyObject_Str(nd); 
  Py_DECREF(nd);
  return retval;
}

/* Representation */
static PyObject* PyBlitzArray_repr(PyBlitzArrayObject* o) {
  return PyString_FromFormat("<blitz.array(%s,%" PY_FORMAT_SIZE_T "d) %" PY_FORMAT_SIZE_T "d elements>", PyBlitzArray_TypenumAsString(o->type_num), o->ndim, PyBlitzArray_len(o));
}

PyTypeObject PyBlitzArray_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                                          /*ob_size*/
    "blitz.bzarray",                            /*tp_name*/
    sizeof(PyBlitzArrayObject),                 /*tp_basicsize*/
    0,                                          /*tp_itemsize*/
    (destructor)PyBlitzArray_Delete,            /*tp_dealloc*/
    0,                                          /*tp_print*/
    0,                                          /*tp_getattr*/
    0,                                          /*tp_setattr*/
    0,                                          /*tp_compare*/
    (reprfunc)PyBlitzArray_repr,                /*tp_repr*/
    0,                                          /*tp_as_number*/
    0,                                          /*tp_as_sequence*/
    &PyBlitzArray_mapping,                      /*tp_as_mapping*/
    0,                                          /*tp_hash */
    0,                                          /*tp_call*/
    (reprfunc)PyBlitzArray_str,                 /*tp_str*/
    0,                                          /*tp_getattro*/
    0,                                          /*tp_setattro*/
    0,                                          /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /*tp_flags*/
    "An N-dimensional blitz::Array<> pythonic representation", /* tp_doc */
    0,		                                      /* tp_traverse */
    0,		                                      /* tp_clear */
    0,		                                      /* tp_richcompare */
    0,		                                      /* tp_weaklistoffset */
    0,		                                      /* tp_iter */
    0,		                                      /* tp_iternext */
    PyBlitzArray_methods,                       /* tp_methods */
    0,                                          /* tp_members */
    PyBlitzArray_getseters,                     /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)PyBlitzArray_init,                /* tp_init */
    0,                                          /* tp_alloc */
    PyBlitzArray_New,                           /* tp_new */
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

  if (PyErr_Occurred()) {
    // we need numpy.ndarray to properly function
    PyErr_Print();
    PyErr_SetString(PyExc_ImportError, "blitz._array failed to import following NumPy import failure");
    return;
  }

  PyBlitzArray_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBlitzArray_Type) < 0) return;

  m = Py_InitModule3("_array", array_methods,
      "blitz::Array<> definition and generic functions");

  Py_INCREF(&PyBlitzArray_Type);
  PyModule_AddObject(m, "array", (PyObject *)&PyBlitzArray_Type);

  /* imports the NumPy C-API as well */
  import_array();
}
