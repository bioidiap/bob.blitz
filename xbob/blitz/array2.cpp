/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue 01 Oct 2013 15:37:07 CEST
 *
 * @brief Pure python bindings for Blitz Arrays
 */

#include "bob/py.h"
#include <memory>

typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  std::shared_ptr<void> bzarr;
  int typenum;
  npy_int  ndim;
  npy_intp shape[4];

} Array;

/**
 * Deallocates memory for an Array object
 */
static void Array_dealloc(Array* self) {
  self->ob_type->tp_free((PyObject*)self);
}

/**
 * Allocates memor and pre-initializes an Array object
 */
static PyObject* Array_new(PyTypeObject* type, PyObject *args, PyObject* kwds) {

  /* Allocates the python object itself */
  Array* self = (Array*)type->tp_alloc(type, 0);

  /* This array is invalid for the time being */
  self->ndim = 0;

  return static_cast<PyObject*>(self);
}

/**
 * Formal initialization of an Array object
 */
static int Array_init(PyTypeObject* type, PyObject *args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static char *kwlist[] = {"shape", "dtype", NULL};

  PyObject* shape;
  PyObject* dtype;

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "OO&", kwlist,
        &shape,
        &PyArray_DescrConverter2, &dtype)
      )
    return -1;

  /* Creates auto-destructable containers for shape and dtype */
  std::shared_ptr<PyObject> ashape = bob::python::new_reference(shape);
  std::shared_ptr<PyObject> adtype = bob::python::new_reference(dtype);

  /* Make sure the dtype is good */
  if (!dtype) {
    PyErr_SetString(PyExc_TypeError, "cannot convert input dtype-like object into a proper dtype");
    return -1;
  }

  /* Make sure shape is good */
  if (!PySequence_Check(shape)) {
    PyErr_SetString(PyExc_TypeError, "shape must be a sequence");
    return -1;
  }

  npy_int ndim = PySequence_Size(shape);
  if (ndim == 0 || ndim > 4) {
    PyErr_Format(PyExc_TypeError, "shape must be a sequence with at least 1 and at most 4 elements (you passed a sequence with %d elements)", ndim);
    return -1;
  }

  /* convert the input information into a shape */
  for (npy_int i=0; i<ndim; ++i) {
    auto item = bob::python::new_reference(PySequence_GetItem(shape, i));
    if (!item) {
      PyErr_Format(PyExc_RuntimeError, "error obtaining element %d from shape sequence", i);
      return -1;
    }
    self->shape[i] = PyLong_AsSsize_t(item.get());
    if (self->shape[i] == -1) {
      PyErr_Format(PyExc_RuntimeError, "error extracting an size from element %d of input shape sequence", i);
    }
  }
  self->ndim = ndim;

  /* at this point we have all data ready for the array creation */
}

static PyTypeObject array2_ArrayType = {
    PyObject_HEAD_INIT(NULL)
    0,                             /*ob_size*/
    "array2.Array",                /*tp_name*/
    sizeof(Array),                 /*tp_basicsize*/
    0,                             /*tp_itemsize*/
    0,                             /*tp_dealloc*/
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
    Py_TPFLAGS_DEFAULT,            /*tp_flags*/
    "An N-dimensional blitz::Array<> pythonic representation", /* tp_doc */
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
