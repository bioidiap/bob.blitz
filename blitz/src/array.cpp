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

PyDoc_STRVAR(s_array_str, "array");

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

  PyErr_Format(PyExc_TypeError, "%s.%s(@%" PY_FORMAT_SIZE_T "d,'%s') indexing requires a single integers (for 1D arrays) or sequences, for any rank size", XSTR(BLITZ_ARRAY_MODULE_PREFIX), s_array_str, self->ndim, PyBlitzArray_TypenumAsString(self->type_num));
  return 0;
}

static int PyBlitzArray_setitem(PyBlitzArrayObject* self, PyObject* item,
    PyObject* value) {

  if (PyNumber_Check(item)) {

    if (self->ndim != 1) {
      PyErr_Format(PyExc_TypeError, "expected sequence for accessing %s.%s(@%" PY_FORMAT_SIZE_T "d,'%s'", XSTR(BLITZ_ARRAY_MODULE_PREFIX), s_array_str, self->ndim, PyBlitzArray_TypenumAsString(self->type_num));
      return -1;
    }

    // if you get to this point, the user has passed single number
    Py_ssize_t k = PyNumber_AsSsize_t(item, PyExc_IndexError);
    return PyBlitzArray_SetItem(self, &k, value);

  }

  if (PySequence_Check(item)) {

    if (self->ndim != PySequence_Fast_GET_SIZE(item)) {
      PyErr_Format(PyExc_TypeError, "expected sequence of size %" PY_FORMAT_SIZE_T "d for accessing %s.%s(@%" PY_FORMAT_SIZE_T "d,'%s')", PySequence_Fast_GET_SIZE(item), XSTR(BLITZ_ARRAY_MODULE_PREFIX), s_array_str, self->ndim, PyBlitzArray_TypenumAsString(self->type_num));
      return -1;
    }

    // if you get to this point, then the input tuple has the same size
    PyBlitzArrayObject shape;
    PyBlitzArrayObject* shape_p = &shape;
    if (!PyBlitzArray_IndexConverter(item, &shape_p)) return 0;
    return PyBlitzArray_SetItem(self, shape.shape, value);

  }

  PyErr_Format(PyExc_TypeError, "%s.%s(@%" PY_FORMAT_SIZE_T "d,'%s') assignment requires a single integers (for 1D arrays) or sequences, for any rank size", XSTR(BLITZ_ARRAY_MODULE_PREFIX), s_array_str, self->ndim, PyBlitzArray_TypenumAsString(self->type_num));
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
numpy.ndarray accessor (shallow wraps " XSTR(BLITZ_ARRAY_MODULE_PREFIX) ".array as numpy.ndarray)"
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
      return PyString_FromFormat("%s.%s(%" PY_FORMAT_SIZE_T "d,'%s')",
          XSTR(BLITZ_ARRAY_MODULE_PREFIX), s_array_str,
          o->shape[0],
          PyBlitzArray_TypenumAsString(o->type_num)
          );
    case 2:
      return PyString_FromFormat("%s.%s((%" PY_FORMAT_SIZE_T "d,%" PY_FORMAT_SIZE_T "d),'%s')",
          XSTR(BLITZ_ARRAY_MODULE_PREFIX), s_array_str,
          o->shape[0],
          o->shape[1],
          PyBlitzArray_TypenumAsString(o->type_num)
          );
    case 3:
      return PyString_FromFormat("%s.%s((%" PY_FORMAT_SIZE_T "d,%" PY_FORMAT_SIZE_T "d,%" PY_FORMAT_SIZE_T "d),'%s')", 
          XSTR(BLITZ_ARRAY_MODULE_PREFIX), s_array_str,
          o->shape[0],
          o->shape[1],
          o->shape[2],
          PyBlitzArray_TypenumAsString(o->type_num)
          );
    case 4:
      return PyString_FromFormat("%s.%s((%" PY_FORMAT_SIZE_T "d,%" PY_FORMAT_SIZE_T "d,%" PY_FORMAT_SIZE_T "d,%" PY_FORMAT_SIZE_T "d),'%s')", 
          XSTR(BLITZ_ARRAY_MODULE_PREFIX), s_array_str,
          o->shape[0],
          o->shape[1],
          o->shape[2],
          o->shape[3],
          PyBlitzArray_TypenumAsString(o->type_num)
          );
    default:
      return PyString_FromFormat("[unsupported] %s.%s(@%" PY_FORMAT_SIZE_T "d,'%s') %" PY_FORMAT_SIZE_T "d elements>",
          XSTR(BLITZ_ARRAY_MODULE_PREFIX), s_array_str,
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

int PyBlitzArray_APIVersion = BLITZ_ARRAY_API_VERSION;

PyTypeObject PyBlitzArray_Type = {
    PyObject_HEAD_INIT(0)
    0,                                          /*ob_size*/
    XSTR(BLITZ_ARRAY_MODULE_PREFIX) ".array",               /*tp_name*/
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

static 
PyObject* PyBlitzArray_as_blitz(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"o", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* retval = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&", kwlist, &PyBlitzArray_Converter, &retval)) return 0;

  return retval;

}

PyDoc_STRVAR(s_as_blitz_str, "as_blitz");
PyDoc_STRVAR(s_as_blitz__doc__,
"as_blitz(x) -> blitz.array\n\
\n\
Converts any compatible python object into a shallow " XSTR(BLITZ_ARRAY_MODULE_PREFIX) ".array\n\
\n\
This function works by first converting the input object ``x`` into\n\
a :py:class:`numpy.ndarray` and then shallow wrapping that ``ndarray``\n\
into a new :py:class:`" XSTR(BLITZ_ARRAY_MODULE_PREFIX) ".array`. You can access the converted\n\
``ndarray`` using the returned value's ``base`` attribute. If the\n\
``ndarray`` cannot be shallow-wrapped, a :py:class:`ValueError` is\n\
raised.\n\
\n\
In the case the input object ``x`` is already a behaved (C-style,\n\
memory-aligned, contiguous) :py:class:`numpy.ndarray`, then this\n\
function only shallow wrap's it into a ``" XSTR(BLITZ_ARRAY_MODULE_PREFIX) ".array`` skin.\n\
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

#define ENTRY_FUNCTION_INNER(a) init ## a
#define ENTRY_FUNCTION(a) ENTRY_FUNCTION_INNER(a)

PyMODINIT_FUNC ENTRY_FUNCTION(BLITZ_ARRAY_MODULE_NAME) (void) {
  PyObject* m;

  PyBlitzArray_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBlitzArray_Type) < 0) return;

  m = Py_InitModule3(XSTR(BLITZ_ARRAY_MODULE_NAME), array_methods,
      "array definition and generic functions");

  /* register version numbers and constants */
  PyModule_AddIntConstant(m, "__api_version__", BLITZ_ARRAY_API_VERSION);
  PyModule_AddStringConstant(m, "__version__", XSTR(BLITZ_ARRAY_VERSION));
  PyModule_AddStringConstant(m, "__numpy_api_name__", XSTR(PY_ARRAY_UNIQUE_SYMBOL));

  /* register the type object to python */
  Py_INCREF(&PyBlitzArray_Type);
  PyModule_AddObject(m, s_array_str, (PyObject *)&PyBlitzArray_Type);

  static void* PyBlitzArray_API[PyBlitzArray_API_pointers];

  /* exhaustive list of C APIs */
  PyBlitzArray_API[PyBlitzArray_APIVersion_NUM] = (void *)&PyBlitzArray_APIVersion;

  // Basic Properties and Checking
  PyBlitzArray_API[PyBlitzArray_Type_NUM] = (void *)&PyBlitzArray_Type;
  PyBlitzArray_API[PyBlitzArray_Check_NUM] = (void *)PyBlitzArray_Check;
  PyBlitzArray_API[PyBlitzArray_CheckNumpyBase_NUM] = (void *)PyBlitzArray_CheckNumpyBase;
  PyBlitzArray_API[PyBlitzArray_TYPE_NUM] = (void *)PyBlitzArray_TYPE;
  PyBlitzArray_API[PyBlitzArray_PyDTYPE_NUM] = (void *)PyBlitzArray_PyDTYPE;
  PyBlitzArray_API[PyBlitzArray_NDIM_NUM] = (void *)PyBlitzArray_NDIM;
  PyBlitzArray_API[PyBlitzArray_SHAPE_NUM] = (void *)PyBlitzArray_SHAPE;
  PyBlitzArray_API[PyBlitzArray_PySHAPE_NUM] = (void *)PyBlitzArray_PySHAPE;
  PyBlitzArray_API[PyBlitzArray_STRIDE_NUM] = (void *)PyBlitzArray_STRIDE;
  PyBlitzArray_API[PyBlitzArray_PySTRIDE_NUM] = (void *)PyBlitzArray_PySTRIDE;
  PyBlitzArray_API[PyBlitzArray_WRITEABLE_NUM] = (void *)PyBlitzArray_WRITEABLE;
  PyBlitzArray_API[PyBlitzArray_PyWRITEABLE_NUM] = (void *)PyBlitzArray_PyWRITEABLE;
  PyBlitzArray_API[PyBlitzArray_BASE_NUM] = (void *)PyBlitzArray_BASE;
  PyBlitzArray_API[PyBlitzArray_PyBASE_NUM] = (void *)PyBlitzArray_PyBASE;

  // Indexing
  PyBlitzArray_API[PyBlitzArray_GetItem_NUM] = (void *)PyBlitzArray_GetItem;
  PyBlitzArray_API[PyBlitzArray_SetItem_NUM] = (void *)PyBlitzArray_SetItem;

  // Construction and Destruction
  PyBlitzArray_API[PyBlitzArray_New_NUM] = (void *)PyBlitzArray_New;
  PyBlitzArray_API[PyBlitzArray_Delete_NUM] = (void *)PyBlitzArray_Delete;
  PyBlitzArray_API[PyBlitzArray_SimpleNew_NUM] = (void *)PyBlitzArray_SimpleNew;
  PyBlitzArray_API[PyBlitzArray_SimpleNewFromData_NUM] = (void *)PyBlitzArray_SimpleNewFromData;

  // From/To NumPy Converters
  PyBlitzArray_API[PyBlitzArray_AsNumpyArray_NUM] = (void *)PyBlitzArray_AsNumpyArray;
  PyBlitzArray_API[PyBlitzArray_FromNumpyArray_NUM] = (void *)PyBlitzArray_FromNumpyArray;
  
  // Converter Functions for PyArg_Parse* family
  PyBlitzArray_API[PyBlitzArray_Converter_NUM] = (void *)PyBlitzArray_Converter;
  PyBlitzArray_API[PyBlitzArray_OutputConverter_NUM] = (void *)PyBlitzArray_OutputConverter;
  PyBlitzArray_API[PyBlitzArray_IndexConverter_NUM] = (void *)PyBlitzArray_IndexConverter;
  PyBlitzArray_API[PyBlitzArray_TypenumConverter_NUM] = (void *)PyBlitzArray_TypenumConverter;

  // Utilities
  PyBlitzArray_API[PyBlitzArray_TypenumAsString_NUM] = (void *)PyBlitzArray_TypenumAsString;

#if PY_VERSION_HEX >= 0x02070000

  /* defines the PyCapsule */

  PyObject* c_api_object = PyCapsule_New((void *)PyBlitzArray_API, 
      XSTR(BLITZ_ARRAY_MODULE_PREFIX) "." XSTR(BLITZ_ARRAY_MODULE_NAME) "._C_API", 0);

#else

  PyObject* c_api_object = PyCObject_FromVoidPtr((void *)PyBlitzArray_API, 0);

#endif

  if (c_api_object) PyModule_AddObject(m, "_C_API", c_api_object);

  /* imports the NumPy C-API as well */
  import_array();
}
