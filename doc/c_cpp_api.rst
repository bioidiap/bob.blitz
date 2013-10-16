.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.dos.anjos@gmail.com>
.. Tue 15 Oct 14:59:05 2013

===============
 C and C++ API
===============

This section includes information for using the pure C or C++ API for
manipulating :py:class:`blitz.array` objects in compiled code.

C API
-----

The C API of ``blitz.array`` allows users to leverage from automatic converters
between :py:class:`numpy.ndarray` and :py:class:`blitz.array` within their own
python extensions. To use the C API, clients should first, include the header
file ``<blitz.array/capi.h>`` on their compilation units and then, make sure to
call once ``import_blitz_array()`` at their module instantiation, as explained
at the `Python manual
<http://docs.python.org/2/extending/extending.html#using-capsules>`_.

Here is a dummy C example showing how to include the header and where to call
the import function:

.. code-block:: c

   #include <blitz.array/capi.h>

   PyMODINIT_FUNC initclient(void) {

     PyObject *m;
 
     m = Py_InitModule("client", ClientMethods);
     if (m == NULL) return;

     if (import_blitz_array() < 0) return;

   }

.. note::

  The include directory can be discovered using :py:func:`blitz.get_include`.

Array Structure
===============

.. c:type:: PyBlitzArrayObject

   The basic array structure represents a ``blitz.array`` instance from the
   C-side of the interpreter:

   .. code-block:: c

      typedef struct {
        PyObject_HEAD
        void* bzarr;
        int type_num;
        Py_ssize_t ndim;
        Py_ssize_t shape[BLITZ_ARRAY_MAXDIMS];
        PyObject* base;

      } PyBlitzArrayObject;

   .. c:macro:: BLITZ_ARRAY_MAXDIMS
      
      The maximum number of dimensions supported by the current ``blitz.array``
      implementation.

   .. c:member:: void* bzarr

      This is a pointer that points to the allocated ``blitz::Array``
      structure. This pointer is cast to the proper type and number of
      dimensions when operations on the data are requested.

   .. c:member:: int type_num

      The numpy type number that is compatible with the elements of this
      array. It is a C representation of the C++ template parameter ``T``.

   .. c:member:: Py_ssize_t ndim

      The rank of the ``blitz::Array<>`` allocated on ``bzarr``.

   .. c:member:: Py_ssize_t shape[BLITZ_ARRAY_MAXDIMS]

      The shape of the ``blitz::Array<>`` allocated on ``bzarr``.

   .. c:member:: PyObject* base

      If the memory pointed by the currently allocated ``blitz::Array<>``
      belongs to another Python object, the object is ``Py_INCREF()``'ed and a
      pointer is kept on this structure member.
   

Accessor Functions
==================

A set of functions allow for creating, deleting, querying and manipulating the
above structure.

.. c:function:: PyObject* PyBlitzArray_AsNumpyArrayCopy (PyBlitzArrayObject* o)
 
   Creates a copy of the given ``blitz.array`` as a ``numpy.ndarray``.
    

.. c:function:: const char* PyBlitzArray_TypenumAsString (int typenum)

   Converts from numpy type_num to a string representation


.. c:function:: PyObject* PyBlitzArray_AsShallowNumpyArray (PyBlitzArrayObject* o)

   Creates a shallow copy of the given ``blitz.array`` as a ``numpy.ndarray``.


.. c:function:: PyObject* PyBlitzArray_SimpleNew (int typenum, Py_ssize_t ndim, Py_ssize_t* shape)

   Allocates a new ``blitz.array`` with a given (supported) type and return it
   as a python object. ``typenum`` should be set to the numpy type number of
   the array type (e.g. ``NPY_FLOAT64``). ``ndim`` should be set to the total
   number of dimensions the array should have. ``shape`` should be set to the
   array shape.
   

.. c:function:: PyObject* PyBlitzArray_GetItem (PyBlitzArrayObject* o, Py_ssize_t* pos)

   Returns, as a PyObject, an item from the array. This will be a copy of the
   internal item. If you set it, it won't set the original array.  ``o`` should
   be the PyBlitzArrayObject to be queried. ``pos`` should be a C-style array
   indicating the precise position to fetch. It is considered to have the same
   number of entries as the current array shape.
   

.. c:function:: int PyBlitzArray_SetItem (PyBlitzArrayObject* o, Py_ssize_t* pos, PyObject* value)

   Sets an given position on the array using any Python or numpy scalar. ``o``
   should be the PyBlitzArrayObject to be set. ``pos`` should be a C-style
   array indicating the precise position to set and ``value``, the Python
   or numpy scalar to set the value to.


.. c:function:: Py_ssize_t PyBlitzArray_NDIM (PyBlitzArrayObject* o)

   Returns the number of dimensions in a given ``blitz.array``.


.. c:function:: int PyBlitzArray_TYPE (PyBlitzArrayObject* o)

   Returns integral type number (as defined by the Numpy C-API) of elements
   in this blitz::Array<>
   

.. c:function:: Py_ssize_t* PyBlitzArray_SHAPE (PyBlitzArrayObject* o)

   Returns the C-stype shape for this blitz::Array<>. This is the formal method
   to query for ``o->shape``.
   

.. c:function:: PyObject* PyBlitzArray_PYSHAPE (PyBlitzArrayObject* o)

   Returns a **new reference** to a Python tuple holding a copy of the shape
   for the given array.
   

.. c:function:: PyArray_Descr* PyBlitzArray_DTYPE (PyBlitzArrayObject* o)

   Returns a **new reference** to a numpy C-API ``PyArray_Descr*`` equivalent
   to the internal type element T.
   

.. c:function:: PyObject* PyBlitzArray_New (PyTypeObject* type, PyObject *args, PyObject* kwds)

   Allocates memory and pre-initializes a ``PyBlitzArrayObject*`` object
   

.. c:function:: void PyBlitzArray_Delete (PyBlitzArrayObject* o)

   Completely deletes a ``PyBlitzArrayObject*`` and associated memory areas.
   

.. c:function:: int PyBlitzArray_IndexConverter (PyObject* o, PyBlitzArrayObject** shape)
 
   Converts any compatible sequence into a C-array containing the shape
   information. The shape information and number of dimensions is stored on
   the previously allocated ``PyBlitzArrayObject*`` you should provide. This
   method is supposed to be used with ``PyArg_ParseTupleAndKeywords`` and
   derivatives.

   Parameters are:
   
   ``o``
     The input object to be converted into a C-shape

   ``shape``
     A preallocated (double) address for storing the shape value, on successful
     conversion
   
   Returns 0 if an error is detected, 1 on success.


.. c:function:: int PyBlitzArray_TypenumConverter (PyObject* o, int** type_num)

   Converts any compatible sequence into a Numpy integer type number. This
   method is supposed to be used with ``PyArg_ParseTupleAndKeywords`` and
   derivatives.

   Parameters are:
   
   ``o``
     The input object to be converted into a C-shape

   ``type_num``
      A preallocated (double) address for storing the type on successful
      conversion.
   
   Returns 0 if an error is detected, 1 on success.
  

.. c:function:: PyObject* PyBlitzArray_AsAnyNumpyArray (PyBlitzArrayObject* o)

   
   Creates a copy of the given ``blitz.array`` as a ``numpy.ndarray`` in the
   most possible efficient way. First try a shallow copy and if that does not
   work, go for a full copy.


.. c:function:: int PyBlitzArray_IsBehaved (PyBlitzArrayObject* o)

  
   Tells if the given ``blitz.array`` can be successfuly wrapped in a shallow
   ``numpy.ndarray``.
  

.. c:function:: int PyBlitzArray_NumpyArrayIsBehaved (PyBlitzArrayObject* o)

   Tells if the given ``numpy.ndarray`` can be successfuly wrapped in a shallow
   ``blitz.array`` (or in a C++ blitz::Array<>) (any will work).


.. c:function:: PyObject* PyBlitzArray_ShallowFromNumpyArray (PyObject* o)

   Creates a new ``blitz.array`` from a ``numpy.ndarray`` object in a shallow
   manner.

C++ API
-------

The C++ API consists mostly of templated methods for manipulating the C++ type
``blitz::Array<>`` so as to convert ``PyObject*``'s from and to objects of that
type. To use the C++ API you must include the header file
``<blitz.array/cppapi.h>`` and ``import_blitz_array()`` on your module, as
explained on the C-API section of this document.

.. cpp:function:: int PyBlitzArray_CToTypenum<T>()

   Converts from C/C++ type to ndarray type_num.
   
   We cover only simple conversions (i.e., standard integers, floats and
   complex numbers only). If the input type is not convertible, an exception
   is set on the Python error stack. You must check ``PyErr_Occurred()`` after
   a call to this function to make sure things are OK and act accordingly.  For
   example:

   .. code-block:: c++
   
      int typenum = PyBlitzArray_CToTypenum<my_weird_type>(obj);
      if (PyErr_Occurred()) return 0; ///< propagate exception


.. cpp:function:: T PyBlitzArray_AsCScalar<T>(PyObject* o)

   Extraction API for **simple** types.
   
   We cover only simple conversions (i.e., standard integers, floats and
   complex numbers only). If the input object is not convertible to the given
   type, an exception is set on the Python error stack. You must check
   ``PyErr_Occurred()`` after a call to this function to make sure things are OK
   and act accordingly. For example:

   .. code-block:: c++
   
      auto z = extract<uint8_t>(obj);
      if (PyErr_Occurred()) return 0; ///< propagate exception
 

.. cpp:function:: blitz::Array<T,N> PyBlitzArray_ShallowFromNumpyArray<T,N>(PyObject* o, bool readwrite)

   Wraps the input numpy ndarray with a blitz::Array<> skin.
   
   You should use this kind of conversion when you either want the ultimate
   speed (as no data copy is involved on this procedure) or when you'd like
   the resulting blitz::Array<> to be writeable, so that you can pass this as
   an input argument to a function and get the results written to the
   original numpy ndarray memory.
   
   Blitz++ is a little more inflexible than numpy ndarrays are, so there are
   limitations in this conversion. For example, normally we can't wrap
   non-contiguous memory areas. In such cases, an exception is set on the
   Python error stack. You must check ``PyErr_Occurred()`` after a call to this
   function to make sure things are OK and act accordingly. For example:

   .. code-block:: c++
   
      auto z = PyBlitzArray_ShallowFromNumpyArray<uint8_t,4>(obj);
      if (PyErr_Occurred()) return 0; ///< propagate exception
   
   Notice that the lifetime of the ``blitz::Array<>`` extracted with this
   procedure is bound to the lifetime of the source numpy ndarray. You'd have
   to copy it to create an independent object.
   

.. cpp:function:: blitz::Array<T,N> PyBlitzArray_FromAny<T,N>(PyObject* o)

   Wraps the input numpy ndarray with a ``blitz::Array<>`` skin, even if it has
   to copy the input data.
   
   You should use this kind of conversion when you only care about finally
   retrieving a ``blitz::Array<>`` of the desired type and shape so as to pass
   it as a const (read-only) input parameter to your C++ method.
   
   At first, we will try a shallow conversion using
   ``PyBlitzArray_AsShallowNumpyArray()`` declared above. If that does not
   work, then we will try a brute force conversion using
   ``PyBlitzArray_AsNumpyArrayCopy()``.  This opens the possibility of, for
   example, converting from objects that support the iteration, buffer, array
   or memory view protocols in python.
   
   Notice that, in this case, the output ``blitz::Array<>`` may or may not be
   bound to the input object. Because you don't know what the outcome is, it is
   recommend you copy the output if you want to preserve it beyond the scope of
   the conversion.
   
   In case of errors, a Python exception will be set. You must check it
   properly:

   .. code-block:: c++
   
      auto z = PyBlitzArray_AsAnyBlitzArray<uint8_t,4>(obj);
      if (PyErr_Occurred()) return 0; ///< propagate exception
   
   Also notice this procedure will copy the data twice, if the input data is
   not already on the right format for a ``blitz::Array<>`` shallow wrap to
   take place. This is not optimal in all conditions, namely with very large
   read-only arrays. We hope this is not a common condition when users want to
   convert read-only arrays.

   
.. cpp:function:: PyObject* PyBlitzArray_AsNumpyArrayCopy<T,N>(const blitz::Array<T,N>& a)

   Copies the contents of the input ``blitz::Array<>`` into a newly allocated
   numpy ndarray.
   
   The newly allocated array is a classical Pythonic **new** reference. The
   client taking the object must call ``Py_DECREF()`` when done.
   
   This function returns ``0`` (null) if an error has occurred, following the
   standard python protocol.
  

.. cpp:function:: PyObject* PyBlitzArray_AsShallowNumpyArray<T,N>(blitz::Array<T,N>& a)

   Creates a read-write shallow copy of the ndarray.
   
   The newly allocated array is a classical Pythonic **new** reference. The
   client taking the object must call ``Py_DECREF()`` when done.
   
   This function returns ``0`` (null) if an error has occurred, following the
   standard python protocol.
  

.. cpp:function:: PyObject* PyBlitzArray_AsAnyNumpyArray<T,N>(blitz::Array<T,N>& a)

   Creates a shallow or copy of the ``blitz::Array<>`` in the fastest possible
   way. Leverages from ``PyBlitzArray_AsShallowNumpyArray`` and
   ``PyBlitzArray_AsNumpyArrayCopy`` as much as possible.
   
   The newly allocated array is a classical Pythonic **new** reference. The
   client taking the object must call ``Py_DECREF()`` when done.
   
   This function returns ``0`` (null) if an error has occurred, following the
   standard python protocol.
 

.. cpp:function:: int PyBlitzArray_IsBehaved<T,N>(blitz::Array<T,N>& a)

   Tells if a shallow wrapping on this ``blitz::Array<>`` would succeed.
