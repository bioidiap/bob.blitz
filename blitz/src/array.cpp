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

PyDoc_STRVAR(s_module_str, "blitz");
PyDoc_STRVAR(s_array_str, "array");

/**
 * Formal initialization of an Array object
 */
static int PyBlitzArray_init(PyBlitzArrayObject* self, PyObject *args,
    PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"shape", "dtype", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

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

  PyErr_Format(PyExc_TypeError, "%s.%s(@%" PY_FORMAT_SIZE_T "d,'%s') indexing requires a single integers (for 1D arrays) or sequences, for any rank size", s_module_str, s_array_str, self->ndim, PyBlitzArray_TypenumAsString(self->type_num));
  return 0;
}

static int PyBlitzArray_setitem(PyBlitzArrayObject* self, PyObject* item,
    PyObject* value) {

  if (PyNumber_Check(item)) {

    if (self->ndim != 1) {
      PyErr_Format(PyExc_TypeError, "expected sequence for accessing %s.%s(@%" PY_FORMAT_SIZE_T "d,'%s'", s_module_str, s_array_str, self->ndim, PyBlitzArray_TypenumAsString(self->type_num));
      return -1;
    }

    // if you get to this point, the user has passed single number
    Py_ssize_t k = PyNumber_AsSsize_t(item, PyExc_IndexError);
    return PyBlitzArray_SetItem(self, &k, value);

  }

  if (PySequence_Check(item)) {

    if (self->ndim != PySequence_Fast_GET_SIZE(item)) {
      PyErr_Format(PyExc_TypeError, "expected sequence of size %" PY_FORMAT_SIZE_T "d for accessing %s.%s(@%" PY_FORMAT_SIZE_T "d,'%s')", PySequence_Fast_GET_SIZE(item), s_module_str, s_array_str, self->ndim, PyBlitzArray_TypenumAsString(self->type_num));
      return -1;
    }

    // if you get to this point, then the input tuple has the same size
    PyBlitzArrayObject shape;
    PyBlitzArrayObject* shape_p = &shape;
    if (!PyBlitzArray_IndexConverter(item, &shape_p)) return 0;
    return PyBlitzArray_SetItem(self, shape.shape, value);

  }

  PyErr_Format(PyExc_TypeError, "%s.%s(@%" PY_FORMAT_SIZE_T "d,'%s') assignment requires a single integers (for 1D arrays) or sequences, for any rank size", s_module_str, s_array_str, self->ndim, PyBlitzArray_TypenumAsString(self->type_num));
  return -1;
}

static PyMappingMethods PyBlitzArray_mapping = {
    (lenfunc)PyBlitzArray_len,
    (binaryfunc)PyBlitzArray_getitem,
    (objobjargproc)PyBlitzArray_setitem,
};

PyDoc_STRVAR(s_private_array_str, "__array__");
PyDoc_STRVAR(s_private_array__doc__,
"x.__array__() -> numpy.ndarray\n\
\n\
numpy.ndarray accessor (choses the fastest possible conversion path)"
);

PyDoc_STRVAR(s_as_ndarray_str, "as_ndarray");
PyDoc_STRVAR(s_as_ndarray__doc__,
"x.as_ndarray() -> numpy.ndarray\n\
\n\
Returns a deep copy as a numpy.ndarray\n\
\n\
This method returns a complete, independent copy of the internal\n\
data as a new :py:class:`numpy.ndarray` of the same data type and\n\
shape. This method should succeed as long as the data type and shape\n\
of current array are supported by the bridge.\n\
"
);

PyDoc_STRVAR(s_as_shallow_ndarray_str, "as_shallow_ndarray");
PyDoc_STRVAR(s_as_shallow_ndarray__doc__,
"x.as_shallow_ndarray() -> numpy.ndarray\n\
\n\
If possible, returns a shallow copy as a :py:class:`numpy.ndarray`\n\
\n\
If this method succeeds, returns a shallow, efficient wrap of the\n\
the internal data. In this case, the ``base`` pointer of the\n\
returned ndarray keeps a pointer to this array to indicate the data\n\
origin. The resulting ndarray object will be read-writeable if this\n\
array also is. Any operation applied to the resulting ndarray will\n\
be reflect on this blitz.array.\n\
\n\
This method will succeed only if the current array type and rank are\n\
supported by our bridge (see array documentation). A\n\
:py:class:`TypeError` is raised otherwise.\n\
"
);

static PyMethodDef PyBlitzArray_methods[] = {
    {
      s_as_ndarray_str,
      (PyCFunction)PyBlitzArray_AsNumpyArrayCopy,
      METH_NOARGS,
      s_as_ndarray__doc__
    },
    {
      s_as_shallow_ndarray_str,
      (PyCFunction)PyBlitzArray_AsShallowNumpyArray,
      METH_NOARGS,
      s_as_shallow_ndarray__doc__
    },
    {
      s_private_array_str,
      (PyCFunction)PyBlitzArray_AsAnyNumpyArray,
      METH_NOARGS,
      s_private_array__doc__
    },
    {0}  /* Sentinel */
};

PyDoc_STRVAR(s_behaved_str, "behaved");
PyDoc_STRVAR(s_behaved__doc__,
"``True``, if the array is aligned and memory contiguous.\n\
``False`` otherwise\n\
"
);

static PyObject* PyBlitzArray_BEHAVED(PyBlitzArrayObject* o) {
  if (PyBlitzArray_IsBehaved(o)) Py_RETURN_TRUE;
  Py_RETURN_FALSE;
}

/* Property API */
PyDoc_STRVAR(s_shape_str, "shape");
PyDoc_STRVAR(s_shape__doc__,
"A tuple indicating the shape of this array"
);

PyDoc_STRVAR(s_dtype_str, "dtype");
PyDoc_STRVAR(s_dtype__doc__,
"The :py:class:`numpy.dtype` for every element in this array"
);

static PyGetSetDef PyBlitzArray_getseters[] = {
    {
      s_dtype_str, 
      (getter)PyBlitzArray_DTYPE,
      0,
      s_dtype__doc__,
      0,
    },
    {
      s_shape_str,
      (getter)PyBlitzArray_PYSHAPE,
      0,
      s_shape__doc__,
      0,
    },
    {
      s_behaved_str,
      (getter)PyBlitzArray_BEHAVED,
      0,
      s_behaved__doc__,
      0,
    },
    {0}  /* Sentinel */
};

/* Stringification */
static PyObject* PyBlitzArray_str(PyBlitzArrayObject* o) {
  PyObject* nd = PyBlitzArray_AsAnyNumpyArray(o);
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
      return PyString_FromFormat("%s.%s(%" PY_FORMAT_SIZE_T "d,'%s')",
          s_module_str, s_array_str,
          o->shape[0],
          PyBlitzArray_TypenumAsString(o->type_num)
          );
    case 2:
      return PyString_FromFormat("%s.%s((%" PY_FORMAT_SIZE_T "d,%" PY_FORMAT_SIZE_T "d),'%s')",
          s_module_str, s_array_str,
          o->shape[0],
          o->shape[1],
          PyBlitzArray_TypenumAsString(o->type_num)
          );
    case 3:
      return PyString_FromFormat("%s.%s((%" PY_FORMAT_SIZE_T "d,%" PY_FORMAT_SIZE_T "d,%" PY_FORMAT_SIZE_T "d),'%s')", 
          s_module_str, s_array_str,
          o->shape[0],
          o->shape[1],
          o->shape[2],
          PyBlitzArray_TypenumAsString(o->type_num)
          );
    case 4:
      return PyString_FromFormat("%s.%s((%" PY_FORMAT_SIZE_T "d,%" PY_FORMAT_SIZE_T "d,%" PY_FORMAT_SIZE_T "d,%" PY_FORMAT_SIZE_T "d),'%s')", 
          s_module_str, s_array_str,
          o->shape[0],
          o->shape[1],
          o->shape[2],
          o->shape[3],
          PyBlitzArray_TypenumAsString(o->type_num)
          );
    default:
      return PyString_FromFormat("[unsupported] %s.%s(@%" PY_FORMAT_SIZE_T "d,'%s') %" PY_FORMAT_SIZE_T "d elements>",
          s_module_str, s_array_str,
          o->ndim,
          PyBlitzArray_TypenumAsString(o->type_num),
          PyBlitzArray_len(o)
          );
  }
}

PyDoc_STRVAR(s_base_str, "base");
PyDoc_STRVAR(s_base__doc__,
"Base object containing the memory this array is pointing to"
);

/* Members */
static PyMemberDef PyBlitzArray_members[] = {
    {
      s_base_str,
      T_OBJECT_EX,
      offsetof(PyBlitzArrayObject, base),
      READONLY,
      s_base__doc__,
    },
    {0}  /* Sentinel */
};

PyDoc_STRVAR(s_array__doc__,
"An N-dimensional blitz::Array<T,N> pythonic representation\n\
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
    "blitz.array",                              /*tp_name*/
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
    (initproc)PyBlitzArray_init,                /* tp_init */
    0,                                          /* tp_alloc */
    PyBlitzArray_New,                           /* tp_new */
};

PyObject* PyBlitzArray_as_blitz(PyObject* self, PyObject* args,
    PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"o", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* arr = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&", kwlist, &PyArray_Converter, &arr)) return 0;

  PyObject* retval = PyBlitzArray_ShallowFromNumpyArray(arr);
  Py_DECREF(arr);
  return retval;
}

PyDoc_STRVAR(s_as_blitz_str, "as_blitz");
PyDoc_STRVAR(s_as_blitz__doc__,
"as_blitz(x) -> blitz.array\n\
\n\
Converts any compatible python object into a blitz.array\n\
\n\
This function works by first converting the input object ``x`` into\n\
a :py:class:`numpy.ndarray` and then shallow wrapping that ``ndarray``\n\
into a new :py:class:`blitz.array`. You can access the converted\n\
``ndarray`` using the returned value's ``base`` attribute.\n\
\n\
In the case the input object ``x`` is already a behaved (C-style,\n\
memory-aligned, contiguous) :py:class:`numpy.ndarray`, then this\n\
function only shallow wrap's it into a ``blitz.array`` skin.\n\
"
);

static PyMethodDef array_methods[] = {
    {
      s_as_blitz_str,
      (PyCFunction)PyBlitzArray_as_blitz,
      METH_VARARGS|METH_KEYWORDS,
      s_as_blitz__doc__
    },
    {0}  /* Sentinel */
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
      "array definition and generic functions");

  Py_INCREF(&PyBlitzArray_Type);
  PyModule_AddObject(m, s_array_str, (PyObject *)&PyBlitzArray_Type);

  /* imports the NumPy C-API as well */
  import_array();
}
