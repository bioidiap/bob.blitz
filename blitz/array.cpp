/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue 01 Oct 2013 15:37:07 CEST
 *
 * @brief Pure python bindings for Blitz Arrays
 */

#define BLITZ_ARRAY_MODULE
#include <blitz.array/capi.h>
#include <structmember.h>

PyDoc_STRVAR(s_array_str, BLITZ_ARRAY_STR(BLITZ_ARRAY_MODULE_PREFIX) ".array");

/**
 * Formal initialization of an Array object
 */
static int PyBlitzArray__init__(PyBlitzArrayObject* self, PyObject *args,
    PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"shape", "dtype", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject shape;
  PyBlitzArrayObject* shape_p = &shape;
  int type_num = NPY_NOTYPE;
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
      PyErr_Format(PyExc_TypeError, "expected tuple for accessing %" PY_FORMAT_SIZE_T "dD array", self->ndim);
      return 0;
    }

    // if you get to this point, the user has passed single number
    Py_ssize_t k = PyNumber_AsSsize_t(item, PyExc_IndexError);
    return PyBlitzArray_GetItem(self, &k);

  }

  if (PySequence_Check(item)) {

    if (self->ndim != PySequence_Fast_GET_SIZE(item)) {
      PyErr_Format(PyExc_TypeError, "expected tuple of size %" PY_FORMAT_SIZE_T "d for accessing %" PY_FORMAT_SIZE_T "dD array", self->ndim, self->ndim);
      return 0;
    }

    // if you get to this point, then the input tuple has the same size
    PyBlitzArrayObject shape;
    PyBlitzArrayObject* shape_p = &shape;
    if (!PyBlitzArray_IndexConverter(item, &shape_p)) return 0;
    return PyBlitzArray_GetItem(self, shape.shape);

  }

  PyErr_Format(PyExc_TypeError, "%s(@%" PY_FORMAT_SIZE_T "d,'%s') indexing requires a single integers (for 1D arrays) or sequences, for any rank size", s_array_str, self->ndim, PyBlitzArray_TypenumAsString(self->type_num));
  return 0;
}

static int PyBlitzArray_setitem(PyBlitzArrayObject* self, PyObject* item,
    PyObject* value) {

  if (PyNumber_Check(item)) {

    if (self->ndim != 1) {
      PyErr_Format(PyExc_TypeError, "expected sequence for accessing %s(@%" PY_FORMAT_SIZE_T "d,'%s'", s_array_str, self->ndim, PyBlitzArray_TypenumAsString(self->type_num));
      return -1;
    }

    // if you get to this point, the user has passed single number
    Py_ssize_t k = PyNumber_AsSsize_t(item, PyExc_IndexError);
    return PyBlitzArray_SetItem(self, &k, value);

  }

  if (PySequence_Check(item)) {

    if (self->ndim != PySequence_Fast_GET_SIZE(item)) {
      PyErr_Format(PyExc_TypeError, "expected sequence of size %" PY_FORMAT_SIZE_T "d for accessing %s(@%" PY_FORMAT_SIZE_T "d,'%s')", PySequence_Fast_GET_SIZE(item), s_array_str, self->ndim, PyBlitzArray_TypenumAsString(self->type_num));
      return -1;
    }

    // if you get to this point, then the input tuple has the same size
    PyBlitzArrayObject shape;
    PyBlitzArrayObject* shape_p = &shape;
    if (!PyBlitzArray_IndexConverter(item, &shape_p)) return 0;
    return PyBlitzArray_SetItem(self, shape.shape, value);

  }

  PyErr_Format(PyExc_TypeError, "%s(@%" PY_FORMAT_SIZE_T "d,'%s') assignment requires a single integers (for 1D arrays) or sequences, for any rank size", s_array_str, self->ndim, PyBlitzArray_TypenumAsString(self->type_num));
  return -1;
}

static PyMappingMethods PyBlitzArray_mapping = {
    (lenfunc)PyBlitzArray_len,
    (binaryfunc)PyBlitzArray_getitem,
    (objobjargproc)PyBlitzArray_setitem,
};

PyDoc_STRVAR(s_as_ndarray_str, "as_ndarray");
PyDoc_STRVAR(s_private_array_str, "__array__");
PyDoc_STRVAR(s_private_array__doc__,
"x.__array__() -> numpy.ndarray\n\
\n\
numpy.ndarray accessor (shallow wraps ``blitz.array`` as numpy.ndarray)"
);

static PyMethodDef PyBlitzArray_methods[] = {
    {
      s_as_ndarray_str,
      (PyCFunction)PyBlitzArray_AsNumpyArray,
      METH_NOARGS,
      s_private_array__doc__
    },
    {
      s_private_array_str,
      (PyCFunction)PyBlitzArray_AsNumpyArray,
      METH_NOARGS,
      s_private_array__doc__
    },
    {0}  /* Sentinel */
};

/* Property API */
PyDoc_STRVAR(s_shape_str, "shape");
PyDoc_STRVAR(s_shape__doc__,
"A tuple indicating the shape of this array (in **elements**)"
);

PyDoc_STRVAR(s_stride_str, "stride");
PyDoc_STRVAR(s_stride__doc__,
"A tuple indicating the strides of this array (in **bytes**)"
);

PyDoc_STRVAR(s_dtype_str, "dtype");
PyDoc_STRVAR(s_dtype__doc__,
"The :py:class:`numpy.dtype` for every element in this array"
);

PyDoc_STRVAR(s_writeable_str, "writeable");
PyDoc_STRVAR(s_writeable__doc__,
"A flag, indicating if this array is writeable"
);

PyDoc_STRVAR(s_base_str, "base");
PyDoc_STRVAR(s_base__doc__,
"If the memory of this array is borrowed from some other object, this is it"
);

static PyGetSetDef PyBlitzArray_getseters[] = {
    {
      s_dtype_str, 
      (getter)PyBlitzArray_PyDTYPE,
      0,
      s_dtype__doc__,
      0,
    },
    {
      s_shape_str,
      (getter)PyBlitzArray_PySHAPE,
      0,
      s_shape__doc__,
      0,
    },
    {
      s_stride_str,
      (getter)PyBlitzArray_PySTRIDE,
      0,
      s_stride__doc__,
      0,
    },
    {
      s_writeable_str,
      (getter)PyBlitzArray_PyWRITEABLE,
      0,
      s_writeable__doc__,
      0,
    },
    {
      s_base_str,
      (getter)PyBlitzArray_PyBASE,
      0,
      s_base__doc__,
      0,
    },
    {0}  /* Sentinel */
};

/* Stringification */
static PyObject* PyBlitzArray_str(PyBlitzArrayObject* o) {
  PyObject* nd = PyBlitzArray_AsNumpyArray(o);
  if (!nd) {
    PyErr_Print();
    PyErr_SetString(PyExc_RuntimeError, "could not convert array into numpy ndarray for str() method call");
    return 0;
  }
  PyObject* retval = PyObject_Str(nd); 
  Py_DECREF(nd);
  return retval;
}

/* Representation */
static PyObject* PyBlitzArray_repr(PyBlitzArrayObject* o) {
  switch (o->ndim) {
    case 1:
      return PyString_FromFormat("%s(%" PY_FORMAT_SIZE_T "d,'%s')",
          s_array_str,
          o->shape[0],
          PyBlitzArray_TypenumAsString(o->type_num)
          );
    case 2:
      return PyString_FromFormat("%s((%" PY_FORMAT_SIZE_T "d,%" PY_FORMAT_SIZE_T "d),'%s')",
          s_array_str,
          o->shape[0],
          o->shape[1],
          PyBlitzArray_TypenumAsString(o->type_num)
          );
    case 3:
      return PyString_FromFormat("%s((%" PY_FORMAT_SIZE_T "d,%" PY_FORMAT_SIZE_T "d,%" PY_FORMAT_SIZE_T "d),'%s')", 
          s_array_str,
          o->shape[0],
          o->shape[1],
          o->shape[2],
          PyBlitzArray_TypenumAsString(o->type_num)
          );
    case 4:
      return PyString_FromFormat("%s((%" PY_FORMAT_SIZE_T "d,%" PY_FORMAT_SIZE_T "d,%" PY_FORMAT_SIZE_T "d,%" PY_FORMAT_SIZE_T "d),'%s')", 
          s_array_str,
          o->shape[0],
          o->shape[1],
          o->shape[2],
          o->shape[3],
          PyBlitzArray_TypenumAsString(o->type_num)
          );
    default:
      return PyString_FromFormat("[unsupported] %s(@%" PY_FORMAT_SIZE_T "d,'%s') %" PY_FORMAT_SIZE_T "d elements>",
          s_array_str,
          o->ndim,
          PyBlitzArray_TypenumAsString(o->type_num),
          PyBlitzArray_len(o)
          );
  }
}

/* Members */
static PyMemberDef PyBlitzArray_members[] = {
    {0}  /* Sentinel */
};

PyDoc_STRVAR(s_array__doc__,
"array(shape, dtype) -> new n-dimensional blitz::Array\n\
\n\
An N-dimensional blitz::Array<T,N> pythonic representation\n\
\n\
Constructor parameters:\n\
\n\
shape\n\
  An iterable, indicating the shape of the array to be constructed\n\
  \n\
  The implementation current supports a maximum of 4 dimensions.\n\
  Building an array with more dimensions will raise a \n\
  :py:class:`TypeError`. There are no explicit limits for the size in\n\
  each dimension, except for the machine's maximum address size.\n\
\n\
dtype\n\
  A :py:class:`numpy.dtype` or ``dtype`` convertible object that\n\
  specified the type of elements in this array.\n\
  \n\
  The following numpy data types are supported by this library:\n\
  \n\
    * :py:class:`numpy.bool_`\n\
    * :py:class:`numpy.int8`\n\
    * :py:class:`numpy.int16`\n\
    * :py:class:`numpy.int32`\n\
    * :py:class:`numpy.int64`\n\
    * :py:class:`numpy.uint8`\n\
    * :py:class:`numpy.uint16`\n\
    * :py:class:`numpy.uint32`\n\
    * :py:class:`numpy.uint64`\n\
    * :py:class:`numpy.float32`\n\
    * :py:class:`numpy.float64`\n\
    * :py:class:`numpy.float128` (if this architecture suppports it)\n\
    * :py:class:`numpy.complex64`\n\
    * :py:class:`numpy.complex128`\n\
    * :py:class:`numpy.complex256` (if this architecture suppports it)\n\
\n\
Objects of this class hold a pointer to C++ ``blitz::Array<T,N>``.\n\
The C++ data type ``T`` is mapped to a :py:class:`numpy.dtype` object,\n\
while the extents and number of dimensions ``N`` are mapped to a shape,\n\
similar to what is done for :py:class:`numpy.ndarray` objects.\n\
\n\
Objects of this class can be wrapped in :py:class:`numpy.ndarray`\n\
quite efficiently, so that flexible numpy-like operations are possible\n\
on its contents. You can also deploy objects of this class wherever\n\
:py:class:`numpy.ndarray`'s may be input.\n\
"
);

PyTypeObject PyBlitzArray_Type = {
    PyObject_HEAD_INIT(0)
    0,                                          /*ob_size*/
    s_array_str,                                /*tp_name*/
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
    s_array__doc__,                             /* tp_doc */
    0,		                                      /* tp_traverse */
    0,		                                      /* tp_clear */
    0,		                                      /* tp_richcompare */
    0,		                                      /* tp_weaklistoffset */
    0,		                                      /* tp_iter */
    0,		                                      /* tp_iternext */
    PyBlitzArray_methods,                       /* tp_methods */
    PyBlitzArray_members,                       /* tp_members */
    PyBlitzArray_getseters,                     /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)PyBlitzArray__init__,             /* tp_init */
    0,                                          /* tp_alloc */
    PyBlitzArray_New,                           /* tp_new */
};
