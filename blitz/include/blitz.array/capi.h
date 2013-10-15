/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue  8 Oct 08:19:28 2013 
 *
 * @brief Defines the blitz.array C-API
 *
 * This module allows somebody else, externally to this package, to include the
 * blitz.array C-API functionality on their own package. Because the API is
 * compiled with a Python module (named `blitz.array`), we need to dig it out
 * from there and bind it to the following C-API members. We do this using a
 * PyCObject/PyCapsule module as explained in:
 * http://docs.python.org/2/extending/extending.html#using-capsules.
 */

#ifndef PY_BLITZARRAY_API_H
#define PY_BLITZARRAY_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>
#include <numpy/arrayobject.h>

/* Maximum number of dimensions supported at this library */
#define BLITZ_ARRAY_MAXDIMS 4

/* Type definition for PyBlitzArrayObject */
typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  void* bzarr; ///< blitz array container
  int type_num; ///< numpy type number of elements
  Py_ssize_t ndim; ///< number of dimensions
  Py_ssize_t shape[BLITZ_ARRAY_MAXDIMS]; ///< shape

  /* Base pointer, if the memory of this object is coming from elsewhere */
  PyObject* base;

} PyBlitzArrayObject;

/* C-API of some Numpy versions we may support */
#define NUMPY17_API 0x00000007
#define NUMPY16_API 0x00000006
#define NUMPY14_API 0x00000004

/*******************
 * C API functions *
 *******************/

#define PyBlitzArray_Type_NUM 0
#define PyBlitzArray_Type_TYPE PyTypeObject

#define PyBlitzArray_AsNumpyArrayCopy_NUM 1
#define PyBlitzArray_AsNumpyArrayCopy_RET PyObject*
#define PyBlitzArray_AsNumpyArrayCopy_PROTO (PyBlitzArrayObject* o)

#define PyBlitzArray_TypenumAsString_NUM 2
#define PyBlitzArray_TypenumAsString_RET const char*
#define PyBlitzArray_TypenumAsString_PROTO (int typenum)

#define PyBlitzArray_AsShallowNumpyArray_NUM 3
#define PyBlitzArray_AsShallowNumpyArray_RET PyObject*
#define PyBlitzArray_AsShallowNumpyArray_PROTO (PyBlitzArrayObject* o)

#define PyBlitzArray_SimpleNew_NUM 4
#define PyBlitzArray_SimpleNew_RET PyObject*
#define PyBlitzArray_SimpleNew_PROTO (int typenum, Py_ssize_t ndim, Py_ssize_t* shape)

#define PyBlitzArray_GetItem_NUM 5
#define PyBlitzArray_GetItem_RET PyObject*
#define PyBlitzArray_GetItem_PROTO (PyBlitzArrayObject* o, Py_ssize_t* pos)

#define PyBlitzArray_SetItem_NUM 6
#define PyBlitzArray_SetItem_RET int
#define PyBlitzArray_SetItem_PROTO (PyBlitzArrayObject* o, Py_ssize_t* pos, PyObject* value)

#define PyBlitzArray_NDIM_NUM 7
#define PyBlitzArray_NDIM_RET Py_ssize_t
#define PyBlitzArray_NDIM_PROTO (PyBlitzArrayObject* o)

#define PyBlitzArray_TYPE_NUM 8
#define PyBlitzArray_TYPE_RET int
#define PyBlitzArray_TYPE_PROTO (PyBlitzArrayObject* o)

#define PyBlitzArray_SHAPE_NUM 9
#define PyBlitzArray_SHAPE_RET Py_ssize_t*
#define PyBlitzArray_SHAPE_PROTO (PyBlitzArrayObject* o)

#define PyBlitzArray_PYSHAPE_NUM 10
#define PyBlitzArray_PYSHAPE_RET PyObject*
#define PyBlitzArray_PYSHAPE_PROTO (PyBlitzArrayObject* o)

#define PyBlitzArray_DTYPE_NUM 11
#define PyBlitzArray_DTYPE_RET PyArray_Descr*
#define PyBlitzArray_DTYPE_PROTO (PyBlitzArrayObject* o)

#define PyBlitzArray_New_NUM 12
#define PyBlitzArray_New_RET PyObject*
#define PyBlitzArray_New_PROTO (PyTypeObject* type, PyObject *args, PyObject* kwds)

#define PyBlitzArray_Delete_NUM 13
#define PyBlitzArray_Delete_RET void
#define PyBlitzArray_Delete_PROTO (PyBlitzArrayObject* o)

#define PyBlitzArray_IndexConverter_NUM 14
#define PyBlitzArray_IndexConverter_RET int
#define PyBlitzArray_IndexConverter_PROTO (PyObject* o, PyBlitzArrayObject** shape)

#define PyBlitzArray_TypenumConverter_NUM 15
#define PyBlitzArray_TypenumConverter_RET int
#define PyBlitzArray_TypenumConverter_PROTO (PyObject* o, int** type_num)

#define PyBlitzArray_AsAnyNumpyArray_NUM 16
#define PyBlitzArray_AsAnyNumpyArray_RET PyObject*
#define PyBlitzArray_AsAnyNumpyArray_PROTO (PyBlitzArrayObject* o)

#define PyBlitzArray_IsBehaved_NUM 17
#define PyBlitzArray_IsBehaved_RET int
#define PyBlitzArray_IsBehaved_PROTO (PyBlitzArrayObject* o)

#define PyBlitzArray_NumpyArrayIsBehaved_NUM 18
#define PyBlitzArray_NumpyArrayIsBehaved_RET int
#define PyBlitzArray_NumpyArrayIsBehaved_PROTO (PyBlitzArrayObject* o)

#define PyBlitzArray_ShallowFromNumpyArray_NUM 19
#define PyBlitzArray_ShallowFromNumpyArray_RET PyObject*
#define PyBlitzArray_ShallowFromNumpyArray_PROTO (PyObject* o)

/* Total number of C API pointers */
#define PyBlitzArray_API_pointers 20

#ifdef BLITZ_ARRAY_MODULE

  /* This section is used when compiling `blitz.array' itself */

  /**
   * The Type object that contains all information necessary to create objects
   * of this type.
   */
  extern PyBlitzArray_Type_TYPE PyBlitzArray_Type;

  PyBlitzArray_AsNumpyArrayCopy_RET PyBlitzArray_AsNumpyArrayCopy PyBlitzArray_AsNumpyArrayCopy_PROTO;

  PyBlitzArray_TypenumAsString_RET PyBlitzArray_TypenumAsString PyBlitzArray_TypenumAsString_PROTO;

  PyBlitzArray_AsShallowNumpyArray_RET PyBlitzArray_AsShallowNumpyArray PyBlitzArray_AsShallowNumpyArray_PROTO;

  PyBlitzArray_SimpleNew_RET PyBlitzArray_SimpleNew PyBlitzArray_SimpleNew_PROTO;

  PyBlitzArray_GetItem_RET PyBlitzArray_GetItem PyBlitzArray_GetItem_PROTO;

  PyBlitzArray_SetItem_RET PyBlitzArray_SetItem PyBlitzArray_SetItem_PROTO;

  PyBlitzArray_NDIM_RET PyBlitzArray_NDIM PyBlitzArray_NDIM_PROTO;

  PyBlitzArray_TYPE_RET PyBlitzArray_TYPE PyBlitzArray_TYPE_PROTO;

  PyBlitzArray_SHAPE_RET PyBlitzArray_SHAPE PyBlitzArray_SHAPE_PROTO;

  PyBlitzArray_PYSHAPE_RET PyBlitzArray_PYSHAPE PyBlitzArray_PYSHAPE_PROTO;

  PyBlitzArray_DTYPE_RET PyBlitzArray_DTYPE PyBlitzArray_DTYPE_PROTO;
  
  PyBlitzArray_New_RET PyBlitzArray_New PyBlitzArray_New_PROTO;

  PyBlitzArray_Delete_RET PyBlitzArray_Delete PyBlitzArray_Delete_PROTO;

  PyBlitzArray_IndexConverter_RET PyBlitzArray_IndexConverter PyBlitzArray_IndexConverter_PROTO;

  PyBlitzArray_TypenumConverter_RET PyBlitzArray_TypenumConverter PyBlitzArray_TypenumConverter_PROTO;

  PyBlitzArray_AsAnyNumpyArray_RET PyBlitzArray_AsAnyNumpyArray PyBlitzArray_AsAnyNumpyArray_PROTO;

  PyBlitzArray_IsBehaved_RET PyBlitzArray_IsBehaved PyBlitzArray_IsBehaved_PROTO;

  PyBlitzArray_NumpyArrayIsBehaved_RET PyBlitzArray_NumpyArrayIsBehaved PyBlitzArray_NumpyArrayIsBehaved_PROTO;

  PyBlitzArray_ShallowFromNumpyArray_RET PyBlitzArray_ShallowFromNumpyArray PyBlitzArray_ShallowFromNumpyArray_PROTO;

#else

  /* This section is used in modules that use `blitz.array's' C-API */

  static void **PyBlitzArray_API;

#define PyBlitzArray_Type (*(PyBlitzArray_Type_TYPE *)PyBlitzArray_API[PyBlitzArray_Type_NUM]

#define PyBlitzArray_System \
  (*(PyBlitzArray_AsNumpyArrayCopy_RET (*)PyBlitzArray_AsNumpyArrayCopy_PROTO) PyBlitzArray_API[PyBlitzArray_AsNumpyArrayCopy_NUM])

#define PyBlitzArray_TypenumAsString \
  (*(PyBlitzArray_TypenumAsString_RET (*)PyBlitzArray_TypenumAsString_PROTO) PyBlitzArray_API[PyBlitzArray_TypenumAsString_NUM])

#define PyBlitzArray_AsShallowNumpyArray \
  (*(PyBlitzArray_AsShallowNumpyArray_RET (*)PyBlitzArray_AsShallowNumpyArray_PROTO) PyBlitzArray_API[PyBlitzArray_AsShallowNumpyArray_NUM])

#define PyBlitzArray_SimpleNew \
  (*(PyBlitzArray_SimpleNew_RET (*)PyBlitzArray_SimpleNew_PROTO) PyBlitzArray_API[PyBlitzArray_SimpleNew_NUM])

#define PyBlitzArray_GetItem \
  (*(PyBlitzArray_GetItem_RET (*)PyBlitzArray_GetItem_PROTO) PyBlitzArray_API[PyBlitzArray_GetItem_NUM])

#define PyBlitzArray_SetItem \
  (*(PyBlitzArray_SetItem_RET (*)PyBlitzArray_SetItem_PROTO) PyBlitzArray_API[PyBlitzArray_SetItem_NUM])

#define PyBlitzArray_NDIM \
  (*(PyBlitzArray_NDIM_RET (*)PyBlitzArray_NDIM_PROTO) PyBlitzArray_API[PyBlitzArray_NDIM_NUM])

#define PyBlitzArray_TYPE \
  (*(PyBlitzArray_TYPE_RET (*)PyBlitzArray_TYPE_PROTO) PyBlitzArray_API[PyBlitzArray_TYPE_NUM])

#define PyBlitzArray_SHAPE \
  (*(PyBlitzArray_SHAPE_RET (*)PyBlitzArray_SHAPE_PROTO) PyBlitzArray_API[PyBlitzArray_SHAPE_NUM])

#define PyBlitzArray_PYSHAPE \
  (*(PyBlitzArray_PYSHAPE_RET (*)PyBlitzArray_PYSHAPE_PROTO) PyBlitzArray_API[PyBlitzArray_PYSHAPE_NUM])

#define PyBlitzArray_DTYPE \
  (*(PyBlitzArray_DTYPE_RET (*)PyBlitzArray_DTYPE_PROTO) PyBlitzArray_API[PyBlitzArray_DTYPE_NUM])

#define PyBlitzArray_New \
  (*(PyBlitzArray_New_RET (*)PyBlitzArray_New_PROTO) PyBlitzArray_API[PyBlitzArray_New_NUM])

#define PyBlitzArray_Delete \
  (*(PyBlitzArray_Delete_RET (*)PyBlitzArray_Delete_PROTO) PyBlitzArray_API[PyBlitzArray_Delete_NUM])

#define PyBlitzArray_IndexConverter \
  (*(PyBlitzArray_IndexConverter_RET (*)PyBlitzArray_IndexConverter_PROTO) PyBlitzArray_API[PyBlitzArray_IndexConverter_NUM])

#define PyBlitzArray_TypenumConverter \
  (*(PyBlitzArray_TypenumConverter_RET (*)PyBlitzArray_TypenumConverter_PROTO) PyBlitzArray_API[PyBlitzArray_TypenumConverter_NUM])

#define PyBlitzArray_AsAnyNumpyArray \
  (*(PyBlitzArray_AsAnyNumpyArray_RET (*)PyBlitzArray_AsAnyNumpyArray_PROTO) PyBlitzArray_API[PyBlitzArray_AsAnyNumpyArray_NUM])

#define PyBlitzArray_IsBehaved \
  (*(PyBlitzArray_IsBehaved_RET (*)PyBlitzArray_IsBehaved_PROTO) PyBlitzArray_API[PyBlitzArray_IsBehaved_NUM])

#define PyBlitzArray_NumpyArrayIsBehaved \
  (*(PyBlitzArray_NumpyArrayIsBehaved_RET (*)PyBlitzArray_NumpyArrayIsBehaved_PROTO) PyBlitzArray_API[PyBlitzArray_NumpyArrayIsBehaved_NUM])

#define PyBlitzArray_ShallowFromNumpyArray \
  (*(PyBlitzArray_ShallowFromNumpyArray_RET (*)PyBlitzArray_ShallowFromNumpyArray_PROTO) PyBlitzArray_API[PyBlitzArray_ShallowFromNumpyArray_NUM])

  /**
   * Returns -1 on error, 0 on success. PyCapsule_Import will set an exception
   * if there's an error.
   */
  static int import_blitz_array(void) {
#if PY_VERSION_HEX >= 0x02070000

    /* New Python API support for library loading */

    PyBlitzArray_API = (void **)PyCapsule_Import("blitz._array", 0);
    return (PyBlitzArray_API != NULL) ? 0 : -1;

#else

    /* Old-style Python API support for library loading */

    PyObject *c_api_object;
    PyObject *module;

    module = PyImport_ImportModule("blitz._array");

    if (module == NULL) return -1;

    c_api_object = PyObject_GetAttrString(module, "_C_API");

    if (c_api_object == NULL) {
      Py_DECREF(module);
      return -1;
    }

    if (PyCObject_Check(c_api_object)) {
      PyBlitzArray_API = (void **)PyCObject_AsVoidPtr(c_api_object);
    }

    Py_DECREF(c_api_object);
    Py_DECREF(module);

    return 0;

#endif
  }

#endif // BLITZ_ARRAY_MODULE

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* PY_BLITZARRAY_API_H */
