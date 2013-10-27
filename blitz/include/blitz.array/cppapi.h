/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 10 Oct 16:02:58 2013 
 *
 * @brief These are C++ extensions in the form of templates that are extras for
 * transforming C++ objects into our Pythonic blitz::Array<> layer.
 */

#ifndef PY_BLITZARRAY_CXXAPI_H
#define PY_BLITZARRAY_CXXAPI_H

#include <blitz.array/capi.h>

#include <complex>
#include <blitz/array.h>
#include <stdint.h>
#include <stdexcept>
#include <typeinfo>

template <typename T> int PyBlitzArrayCxx_CToTypenum() {
  PyErr_Format(PyExc_NotImplementedError, "c++ type to numpy type_num conversion unsupported for typeid.name() `%s'", typeid(T).name());
  return -1;
}

template <> int PyBlitzArrayCxx_CToTypenum<bool>() 
{ return NPY_BOOL; }

template <> int PyBlitzArrayCxx_CToTypenum<int8_t>() 
{ return NPY_INT8; }

template <> int PyBlitzArrayCxx_CToTypenum<uint8_t>() 
{ return NPY_UINT8; }

template <> int PyBlitzArrayCxx_CToTypenum<int16_t>() 
{ return NPY_INT16; }

template <> int PyBlitzArrayCxx_CToTypenum<uint16_t>() 
{ return NPY_UINT16; }

template <> int PyBlitzArrayCxx_CToTypenum<int32_t>() 
{ return NPY_INT32; }

template <> int PyBlitzArrayCxx_CToTypenum<uint32_t>() 
{ return NPY_UINT32; }

template <> int PyBlitzArrayCxx_CToTypenum<int64_t>() 
{ return NPY_INT64; }

template <> int PyBlitzArrayCxx_CToTypenum<uint64_t>() 
{ return NPY_UINT64; }

template <> int PyBlitzArrayCxx_CToTypenum<float>() 
{ return NPY_FLOAT32; }

template <> int PyBlitzArrayCxx_CToTypenum<double>() 
{ return NPY_FLOAT64; }

#ifdef NPY_FLOAT128
template <> int PyBlitzArrayCxx_CToTypenum<long double>() 
{ return NPY_FLOAT128; }
#endif

template <> int PyBlitzArrayCxx_CToTypenum<std::complex<float>>() 
{ return NPY_COMPLEX64; }

template <> int PyBlitzArrayCxx_CToTypenum<std::complex<double>>() 
{ return NPY_COMPLEX128; }

#ifdef NPY_COMPLEX256
template <> int PyBlitzArrayCxx_CToTypenum<std::complex<long double>>() 
{ return NPY_COMPLEX256; }
#endif

#ifdef __APPLE__
template <> int PyBlitzArrayCxx_CToTypenum<long>() {
  if (sizeof(long) == 4) return NPY_INT32;
  return NPY_INT64;
}

template <> int PyBlitzArrayCxx_CToTypenum<unsigned long>() {
  if (sizeof(unsigned long) == 4) return NPY_UINT32;
  return NPY_UINT64;
}
#endif

template <typename T> T PyBlitzArrayCxx_AsCScalar(PyObject* o) {

  int type_num = PyBlitzArrayCxx_CToTypenum<T>();
  if (PyErr_Occurred()) {
    T retval = 0;
    return retval;
  }

  // create a zero-dimensional array on the expected type
  PyArrayObject* zerodim =
    reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(0, 0, type_num));

  if (!zerodim) {
    T retval = 0;
    return retval;
  }

  int status = PyArray_SETITEM(zerodim,
      reinterpret_cast<char*>(PyArray_DATA(zerodim)), o);

  if (status != 0) {
    T retval = 0;
    return retval;
  }

  // note: this will decref `zerodim'
  PyObject* scalar=PyArray_Return(zerodim);

  if (!scalar) {
    T retval = 0;
    return retval;
  }

  T retval = 0;
  PyArray_ScalarAsCtype(scalar, &retval);
  Py_DECREF(scalar);
  return retval;
}

template <typename T, int N> 
int PyBlitzArrayCxx_IsBehaved(const blitz::Array<T,N>& a) {

  if(!a.isStorageContiguous()) return 0;

  for(int i=0; i<a.rank(); ++i) {
    if(!(a.isRankStoredAscending(i) && a.ordering(i)==a.rank()-1-i))
      return 0;
  }

  //if you get to this point, nothing else to-do rather than return true
  return 1;
}

template <typename T, int N> 
PyObject* PyBlitzArrayCxx_NewFromConstArray(const blitz::Array<T,N>& a) {

  if (!PyBlitzArrayCxx_IsBehaved(a)) {
    PyErr_Format(PyExc_ValueError, "cannot convert C++ blitz::Array<%s,%d> which doesn't behave (memory contiguous, aligned, C-style) into a pythonic blitz.array", PyBlitzArray_TypenumAsString(PyBlitzArrayCxx_CToTypenum<T>()), N);
    return 0;
  }

  try {

    PyTypeObject& tp = PyBlitzArray_Type;
    PyBlitzArrayObject* retval = (PyBlitzArrayObject*)PyBlitzArray_New(&tp, 0, 0);
    retval->bzarr = static_cast<void*>(new blitz::Array<T,N>(a));
    retval->data = const_cast<void*>(static_cast<const void*>(a.data()));
    retval->type_num = PyBlitzArrayCxx_CToTypenum<T>();
    retval->ndim = N;
    for (Py_ssize_t i=0; i<N; ++i) {
      retval->shape[i] = a.extent(i);
      retval->stride[i] = sizeof(T)*a.stride(i); ///in **bytes**
    }
    retval->writeable = 0;
    return reinterpret_cast<PyObject*>(retval);

  }

  catch (std::exception& e) {
    PyErr_Format(PyExc_RuntimeError, "caught exception while instantiating blitz.array(@%" PY_FORMAT_SIZE_T "d,'%s'): %s", N, PyBlitzArray_TypenumAsString(PyBlitzArrayCxx_CToTypenum<T>()), e.what());
  }

  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "caught unknown exception while instantiating blitz.array(@%" PY_FORMAT_SIZE_T "d,'%s')", N, PyBlitzArray_TypenumAsString(PyBlitzArrayCxx_CToTypenum<T>()));
  }

  /** some test code
  std::cout << "allocating array" << std::endl;
  std::shared_ptr<blitz::Array<T,N>> retval(new blitz::Array<T,N>(tv_shape),
      &delete_array<T,N>);
  **/

  return 0;

}

template<typename T, int N>
PyObject* PyBlitzArrayCxx_NewFromArray(blitz::Array<T,N>& a) {

  PyObject* retval = PyBlitzArrayCxx_NewFromConstArray(a);

  if (!retval) return retval;

  reinterpret_cast<PyBlitzArrayObject*>(retval)->writeable = 1;

  return retval;

}

template<typename T, int N>
blitz::Array<T,N>* PyBlitzArrayCxx_AsBlitz(PyBlitzArrayObject* o) {
  return reinterpret_cast<blitz::Array<T,N>*>(o->bzarr);
}

#endif /* PY_BLITZARRAY_CXXAPI_H */
