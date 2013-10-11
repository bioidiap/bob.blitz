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
  int type_num; ///< numpy type number of elements on `bzarr'
  Py_ssize_t ndim; ///< number of dimensions of `bzarr'
  Py_ssize_t shape[BLITZ_ARRAY_MAXDIMS]; ///< shape of `bzarr'

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

  /**
   * Creates a copy of the given blitz::Array<> as a Numpy ndarray.
   * 
   * @param typenum The numpy type number of the array type
   * @param ndim The total number of dimensions
   * @param bz The pre-allocated array
   */
  PyBlitzArray_AsNumpyArrayCopy_RET PyBlitzArray_AsNumpyArrayCopy PyBlitzArray_AsNumpyArrayCopy_PROTO;

  /**
   * Converts from numpy type_num to a string representation
   */
  PyBlitzArray_TypenumAsString_RET PyBlitzArray_TypenumAsString PyBlitzArray_TypenumAsString_PROTO;

  /**
   * Creates a shallow copy of the given blitz::Array<> as a Numpy ndarray.
   * 
   * @param typenum The numpy type number of the array type
   * @param ndim The total number of dimensions
   * @param bz The pre-allocated array
   */
  PyBlitzArray_AsShallowNumpyArray_RET PyBlitzArray_AsShallowNumpyArray PyBlitzArray_AsShallowNumpyArray_PROTO;

  /**
   * Allocates a blitz::Array<> with a given (supported) type and return it as
   * a smart void*.
   *
   * @param typenum The numpy type number of the array type
   * @param ndim The total number of dimensions
   * @param shape The array shape
   */
  PyBlitzArray_SimpleNew_RET PyBlitzArray_SimpleNew PyBlitzArray_SimpleNew_PROTO;

  /**
   * Returns, as a PyObject, an item from the array. This will be a copy of the
   * internal item. If you set it, it won't set the original array.
   *
   * @param o The PyBlitzArrayObject to be queried
   * @param pos An array indicating the precise position to fetch. It is
   * considered to have the same number of entries as the current array shape.
   */
  PyBlitzArray_GetItem_RET PyBlitzArray_GetItem PyBlitzArray_GetItem_PROTO;

  /**
   * Sets an given position on the array using any Python or numpy scalar.
   *
   * @param o The PyBlitzArrayObject to be set
   * @param pos An array indicating the precise position to fetch
   * @param value The Python scalar to set the value to
   */
  PyBlitzArray_SetItem_RET PyBlitzArray_SetItem PyBlitzArray_SetItem_PROTO;

  /**
   * Returns the number of dimensions in a given blitz::Array<>
   *
   * @param o The input PyBlitzArray to be queried
   */
  PyBlitzArray_NDIM_RET PyBlitzArray_NDIM PyBlitzArray_NDIM_PROTO;

  /**
   * Returns integral type number (as defined by the Numpy C-API) of elements
   * in this blitz::Array<>
   *
   * @param o The input PyBlitzArray to be queried
   */
  PyBlitzArray_TYPE_RET PyBlitzArray_TYPE PyBlitzArray_TYPE_PROTO;

  /**
   * Returns the C-stype shape for this blitz::Array<> 
   *
   * @param o The input PyBlitzArray to be queried
   */
  PyBlitzArray_SHAPE_RET PyBlitzArray_SHAPE PyBlitzArray_SHAPE_PROTO;

  /**
   * Returns a <b>new reference</b> to a Python tuple holding a copy of shape
   * for this blitz::Array<>
   *
   * @param o The input PyBlitzArray to be queried
   */
  PyBlitzArray_PYSHAPE_RET PyBlitzArray_PYSHAPE PyBlitzArray_PYSHAPE_PROTO;

  /**
   * Returns a <b>new reference</b> to a numpy C-API PyArray_Descr*
   * equivalent to the internal type element T.
   *
   * @param o The input PyBlitzArray to be queried
   */
  PyBlitzArray_DTYPE_RET PyBlitzArray_DTYPE PyBlitzArray_DTYPE_PROTO;
  
  /**
   * Allocates memory and pre-initializes a PyBlitzArrayObject object
   */
  PyBlitzArray_New_RET PyBlitzArray_New PyBlitzArray_New_PROTO;

  /**
   * Completely deletes a PyBlitzArrayObject* and associated memory areas.
   *
   * @param o The input PyBlitzArray to be deleted
   */
  PyBlitzArray_Delete_RET PyBlitzArray_Delete PyBlitzArray_Delete_PROTO;

  /**
   * Converts any compatible sequence into a C-array containing the shape
   * information. The shape information and number of dimensions is stored on
   * the previously allocated PyBlitzArrayObject* you should provide. This
   * method is supposed to be used with PyArg_ParseTupleAndKeywords and
   * derivatives.
   *
   * @param o The input object to be converted into a C-shape
   * @param shape A preallocated (double) address for storing the shape value,
   *        on successful conversion
   *
   * Returns 0 if an error is detected, 1 on success.
   */
  PyBlitzArray_IndexConverter_RET PyBlitzArray_IndexConverter PyBlitzArray_IndexConverter_PROTO;

  /**
   * Converts any compatible sequence into a Numpy integer type number. This
   * method is supposed to be used with PyArg_ParseTupleAndKeywords and
   * derivatives.
   *
   * @param o The input object to be converted into a C-shape
   * @param type_num A preallocated (double) address for storing the type on
   *                 successful conversion.
   *
   * Returns 0 if an error is detected, 1 on success.
   */
  PyBlitzArray_TypenumConverter_RET PyBlitzArray_TypenumConverter PyBlitzArray_TypenumConverter_PROTO;

  /**
   * Creates a copy of the given blitz::Array<> as a Numpy ndarray in the most
   * possible efficient way. First try a shallow copy and if that does not
   * work, go for a full copy.
   * 
   * @param o The blitz::Array<> to efficiently copy
   */
  PyBlitzArray_AsAnyNumpyArray_RET PyBlitzArray_AsAnyNumpyArray PyBlitzArray_AsAnyNumpyArray_PROTO;

  /**
   * Tells if the given blitz::Array<> can be successfuly wrapped in a shallow
   * numpy.ndarray.
   * 
   * @param o The blitz::Array<> to check
   */
  PyBlitzArray_IsBehaved_RET PyBlitzArray_IsBehaved PyBlitzArray_IsBehaved_PROTO;

  /**
   * Tells if the given numpy.ndarray can be successfuly wrapped in a shallow
   * blitz.array or in a C++ blitz::Array<> (any will work).
   * 
   * @param o The numpy ndarray to check
   */
  PyBlitzArray_NumpyArrayIsBehaved_RET PyBlitzArray_NumpyArrayIsBehaved PyBlitzArray_NumpyArrayIsBehaved_PROTO;

  /**
   * Creates a new PyBlitzArrayObject from a Numpy ndarray object in a shallow
   * manner.
   * 
   * @param o The numpy ndarray to shallow copy
   */
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
