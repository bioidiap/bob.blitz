/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 10 Oct 16:02:58 2013 
 *
 * @brief These are C++ extensions in the form of templates that are extras for
 * transforming C++ objects into our Pythonic blitz::Array<> layer.
 */

#include <blitz.array/capi.h>

#include <complex>
#include <blitz/array.h>
#include <stdint.h>
#include <stdexcept>
#include <typeinfo>
#include <memory>

/**
 * @brief Converts from C/C++ type to ndarray type_num.
 *
 * We cover only simple conversions (i.e., standard integers, floats and
 * complex numbers only). If the input type is not convertible, an exception
 * is set on the Python error stack. You must check `PyErr_Occurred()` after
 * a call to this function to make sure things are OK and act accordingly.
 * For example:
 *
 * int typenum = PyBlitzArray_CToTypenum<my_weird_type>(obj);
 * if (PyErr_Occurred()) return 0; ///< propagate exception
 *
 * Note: This generic implementation only raises an "NotImplementedError"
 * exception in Python.
 */
template <typename T> int PyBlitzArray_CToTypenum() {
  PyErr_Format(PyExc_NotImplementedError, "c++ type to numpy type_num conversion unsupported for typeid.name() `%s'", typeid(T).name());
  return -1;
}

template <> int PyBlitzArray_CToTypenum<bool>() 
{ return NPY_BOOL; }

template <> int PyBlitzArray_CToTypenum<int8_t>() 
{ return NPY_INT8; }

template <> int PyBlitzArray_CToTypenum<uint8_t>() 
{ return NPY_UINT8; }

template <> int PyBlitzArray_CToTypenum<int16_t>() 
{ return NPY_INT16; }

template <> int PyBlitzArray_CToTypenum<uint16_t>() 
{ return NPY_UINT16; }

template <> int PyBlitzArray_CToTypenum<int32_t>() 
{ return NPY_INT32; }

template <> int PyBlitzArray_CToTypenum<uint32_t>() 
{ return NPY_UINT32; }

template <> int PyBlitzArray_CToTypenum<int64_t>() 
{ return NPY_INT64; }

template <> int PyBlitzArray_CToTypenum<uint64_t>() 
{ return NPY_UINT64; }

template <> int PyBlitzArray_CToTypenum<float>() 
{ return NPY_FLOAT32; }

template <> int PyBlitzArray_CToTypenum<double>() 
{ return NPY_FLOAT64; }

#ifdef NPY_FLOAT128
template <> int PyBlitzArray_CToTypenum<long double>() 
{ return NPY_FLOAT128; }
#endif

template <> int PyBlitzArray_CToTypenum<std::complex<float>>() 
{ return NPY_COMPLEX64; }

template <> int PyBlitzArray_CToTypenum<std::complex<double>>() 
{ return NPY_COMPLEX128; }

#ifdef NPY_COMPLEX256
template <> int PyBlitzArray_CToTypenum<std::complex<long double>>() 
{ return NPY_COMPLEX256; }
#endif

#ifdef __APPLE__
template <> int PyBlitzArray_CToTypenum<long>() {
  if (sizeof(long) == 4) return NPY_INT32;
  return NPY_INT64;
}

template <> int PyBlitzArray_CToTypenum<unsigned long>() {
  if (sizeof(unsigned long) == 4) return NPY_UINT32;
  return NPY_UINT64;
}
#endif

/**
 * @brief Extraction API for **simple** types.
 *
 * We cover only simple conversions (i.e., standard integers, floats and
 * complex numbers only). If the input object is not convertible to the given
 * type, an exception is set on the Python error stack. You must check
 * `PyErr_Occurred()` after a call to this function to make sure things are
 * OK and act accordingly. For example:
 *
 * auto z = extract<uint8_t>(obj);
 * if (PyErr_Occurred()) return 0; ///< propagate exception
 */
template <typename T> T PyBlitzArray_AsCScalar(PyObject* o) {

  int type_num = PyBlitzArray_CToTypenum<T>();
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

/**
 * @brief Wraps the input numpy ndarray with a blitz::Array<> skin.
 *
 * You should use this kind of conversion when you either want the ultimate
 * speed (as no data copy is involved on this procedure) or when you'd like
 * the resulting blitz::Array<> to be writeable, so that you can pass this as
 * an input argument to a function and get the results written to the
 * original numpy ndarray memory.
 *
 * Blitz++ is a little more inflexible than numpy ndarrays are, so there are
 * limitations in this conversion. For example, normally we can't wrap
 * non-contiguous memory areas. In such cases, an exception is set on the
 * Python error stack. You must check `PyErr_Occurred()` after a call to this
 * function to make sure things are OK and act accordingly. For example:
 *
 * auto z = shallow_blitz_array<uint8_t,4>(obj);
 * if (PyErr_Occurred()) return 0; ///< propagate exception
 *
 * Notice that the lifetime of the blitz::Array<> extracted with this
 * procedure is bound to the lifetime of the source numpy ndarray. You'd have
 * to copy it to create an independent object.
 */
template <typename T, int N> blitz::Array<T,N> PyBlitzArray_AsShallowBlitzArray
(PyObject* o, bool readwrite) {

  int type_num = PyBlitzArray_CToTypenum<T>();
  if (PyErr_Occurred()) {
    // propagates exception
    blitz::Array<T,N> retval;
    return retval;
  }

  PyArray_Descr* req_dtype = PyArray_DescrFromType(type_num); ///< borrowed

  // has to be an ndarray
  if (!PyArray_Check(o)) {
    PyErr_Format(PyExc_TypeError, "object is not of type numpy.ndarray and cannot be wrapped as shallow blitz::Array<%s,%d>", type_num, N);
    blitz::Array<T,N> retval;
    return retval;
  }

  PyArrayObject* ao = reinterpret_cast<PyArrayObject*>(o);

  // checks that the data type matches
  if (!PyArray_EquivTypes(PyArray_DESCR(ao), req_dtype)) {
    PyErr_Format(PyExc_TypeError, "numpy.ndarray has data type `%c%d' and cannot be wrapped as shallow blitz::Array<%s,%d>", PyArray_DESCR(ao)->kind, PyArray_DESCR(ao)->elsize, PyBlitzArray_TypenumAsString(type_num), N);
    blitz::Array<T,N> retval;
    return retval;
  }

  // checks that the shape matches
  if (PyArray_NDIM(ao) != N) {
    PyErr_Format(PyExc_TypeError, "numpy.ndarray has %d dimensions and cannot be wrapped as shallow blitz::Array<%s,%d>", PyArray_NDIM(ao), PyBlitzArray_TypenumAsString(type_num), N);
    blitz::Array<T,N> retval;
    return retval;
  }

  // checks for writeability
  if (readwrite && !PyArray_ISWRITEABLE(ao)) {
    PyErr_Format(PyExc_TypeError, "numpy.ndarray is not writeable and cannot be wrapped as a read-writeable blitz::Array<%s,%d>", PyBlitzArray_TypenumAsString(type_num), N);
    blitz::Array<T,N> retval;
    return retval;
  }

  // checks if the array is C-style, aligned and in contiguous memory
  if (!PyArray_ISCARRAY_RO(ao)) {
    PyErr_Format(PyExc_TypeError, "numpy.ndarray does not behave (C-style, memory aligned and contiguous) and cannot be wrapped as shallow blitz::Array<%s,%d>", PyBlitzArray_TypenumAsString(type_num), N);
    blitz::Array<T,N> retval;
    return retval;
  }

  // if you survived to this point, do the real work
  blitz::TinyVector<int,N> shape;
  blitz::TinyVector<int,N> stride;
  for (int i=0; i<N; ++i) {
    shape[i] = PyArray_DIMS(ao)[i];
    stride[i] = PyArray_STRIDES(ao)[i];
  }
  return blitz::Array<T,N>(static_cast<T*>(PyArray_DATA(ao)),
      shape, stride, blitz::neverDeleteData);
}

/**
 * @brief Wraps the input numpy ndarray with a blitz::Array<> skin, even if
 * it has to copy the input data.
 *
 * You should use this kind of conversion when you only care about finally
 * retrieving a blitz::Array<> of the desired type and shape so as to pass it
 * as a const (read-only) input parameter to your C++ method.
 *
 * At first, we will try a shallow conversion using `shallow_blitz_array()`
 * declared above. If that does not work, then we will try a brute force
 * conversion using `PyArray_FromAny()`. This opens the possibility of, for
 * example, converting from objects that support the iteration, buffer, array
 * or memory view protocols in python.
 *
 * Notice that, in this case, the output blitz::Array may or may not be bound
 * to the input object. Because you don't know what the outcome is, it is
 * recommend you copy the output if you want to preserve it beyond the scope
 * of the conversion.
 *
 * In case of errors, a Python exception will be set. You must check it
 * properly:
 *
 * auto z = readonly_blitz_array<uint8_t,4>(obj);
 * if (PyErr_Occurred()) return 0; ///< propagate exception
 *
 * Notice that the lifetime of the blitz::Array<> extracted with this
 * procedure is bound to the lifetime of the source numpy ndarray. You'd have
 * to copy it to create an independent object.
 *
 * Also notice this procedure will copy the data twice, if the input data is
 * not already on the right format for a blitz::Array<> shallow wrap to take
 * place. This is not optimal in all conditions, namely with very large
 * read-only arrays. We hope this is not a common condition when users want
 * to convert read-only arrays.
 */
template <typename T, int N> blitz::Array<T,N> PyBlitzArray_AsAnyBlitzArray
(PyObject* o) {

  blitz::Array<T,N> shallow = PyBlitzArray_AsShallowBlitzArray<T,N>(o, false);
  if (!PyErr_Occurred()) return shallow;

  // if you get to this point, than shallow conversion did not work

  int type_num = PyBlitzArray_CToTypenum<T>();
  if (PyErr_Occurred()) {
    // propagates exception
    blitz::Array<T,N> retval;
    return retval;
  }

  PyArray_Descr* req_dtype = PyArray_DescrFromType(type_num); ///< borrowed

  // transforms the data and wraps with an auto-deletable object
  PyObject* newref = PyArray_FromAny(o, req_dtype, N, N,
#     if NPY_FEATURE_VERSION >= NUMPY17_API /* NumPy C-API version >= 1.7 */
      NPY_ARRAY_CARRAY_RO,
#     else
      NPY_CARRAY_RO,
#     endif
      0);

  if (!newref) {
    // propagates exception
    blitz::Array<T,N> retval;
    return retval;
  }

  // wrap the new array with a shallow skin
  blitz::Array<T,N> bz_shallow = PyBlitzArray_AsShallowBlitzArray<T,N>(newref, false);
  if (!PyErr_Occurred()) {
    Py_DECREF(newref);
    blitz::Array<T,N> retval;
    return retval;
  }

  // copies and returns - do not use "copy" - this is the SECOND copy
  blitz::Array<T,N> copy(bz_shallow.shape());
  copy = bz_shallow;
  Py_DECREF(newref);
  return copy;
}

/**
 * @brief Copies the contents of the input blitz::Array<> into a newly
 * allocated numpy ndarray.
 *
 * The newly allocated array is a classical Pythonic **new** reference. The
 * client taking the object must call Py_XDECREF when done.
 *
 * This function returns NULL if an error has occurred, following the
 * standard python protocol.
 */
template <typename T, int N>
PyObject* PyBlitzArray_AsNumpyNDArrayCopy(const blitz::Array<T,N>& a) {

  int type_num = PyBlitzArray_CToTypenum<T>();
  if (PyErr_Occurred()) return 0;

  // maximum supported number of dimensions
  if (N > 11) {
    PyErr_Format(PyExc_TypeError, "input blitz::Array<%s,%d> has more dimensions than we can support (max. = 11)", PyBlitzArray_TypenumAsString(type_num), N);
    return 0;
  }

  // copies the shape
  npy_intp shape[NPY_MAXDIMS];
  for (int i=0; i<N; ++i) shape[i] = a.extent(i);

  // creates a new array that will be returned
  PyObject* retval = PyArray_SimpleNew(N, shape, type_num);
  if (!retval) return 0; ///< propagate exception

  // creates a simple blitz::Array wrapper around the ndarray
  blitz::TinyVector<int,N> bz_shape;
  for (int i=0; i<N; ++i) bz_shape[i] = shape[i];
  PyArrayObject* aret = reinterpret_cast<PyArrayObject*>(retval);
  blitz::Array<T,N> shallow(static_cast<T*>(PyArray_DATA(aret)),
      bz_shape, blitz::neverDeleteData);

  // copies the data from the source array to the shallow blitz array
  shallow = a;

  return retval;

}

/**
 * @brief Creates a **readonly** shallow copy of the ndarray.
 *
 * The newly allocated array is a classical Pythonic **new** reference. The
 * client taking the object must call Py_XDECREF when done.
 *
 * This function returns NULL if an error has occurred, following the
 * standard python protocol.
 */
template <typename T, int N>
PyObject* PyBlitzArray_AsShallowNumpyNDArray(blitz::Array<T,N>& a) {

  int type_num = PyBlitzArray_CToTypenum<T>();
  if (PyErr_Occurred()) return 0;

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

  // maximum supported number of dimensions
  if (N > 11) {
    PyErr_Format(PyExc_TypeError, "input blitz::Array<%s,%d> has more dimensions than we can support (max. = 11)", PyBlitzArray_TypenumAsString(type_num), N);
    return 0;
  }

  // copies the shape
  npy_intp shape[NPY_MAXDIMS];
  for (int i=0; i<N; ++i) shape[i] = a.extent(i);

  // creates an ndarray from the blitz::Array<>.data()
  return PyArray_NewFromDescr(&PyArray_Type, 
      PyArray_DescrFromType(type_num),
      N, shape, 0, reinterpret_cast<void*>(a.data()),
#     if NPY_FEATURE_VERSION >= NUMPY17_API /* NumPy C-API version >= 1.7 */
      NPY_ARRAY_BEHAVED,
#     else
      NPY_BEHAVED,
#     endif
      0);

}
