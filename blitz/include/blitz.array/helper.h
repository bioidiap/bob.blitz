/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 12 Sep 08:16:31 2013
 *
 * @brief Utility classes and functions for conversion between Blitz C++
 * interface and Python's C-API.
 */

#ifndef PY_BLITZ_ARRAY_HELPER_H
#define PY_BLITZ_ARRAY_HELPER_H

#include <blitz.array/api.h>

#include <memory>

#define NUMPY17_API 0x00000007
#define NUMPY16_API 0x00000006
#define NUMPY14_API 0x00000004

namespace pybz { namespace detail {

  /**
   * Returns a std::shared_ptr that wraps a PyObject and will Py_XDECREF'it
   * when gone.
   */
  std::shared_ptr<PyObject> handle(PyObject* o);

  /**
   * Returns a std::shared_ptr that wraps a PyObject and will NOT Py_XDECREF'it
   * when gone.
   */
  std::shared_ptr<PyObject> borrowed(PyObject* o);

  /**
   * Safely creates a new reference to an existing PyObject
   */
  PyObject* new_reference(std::shared_ptr<PyObject> o);

}}

#endif /* PYBZ_ARRAY_HELPER_H */
