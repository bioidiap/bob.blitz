/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date Thu 21 May 12:31:49 CEST 2015
 *
 * @brief Functionality to provide version information in the python bindings
 */

#ifndef BOB_BLITZ_CONFIG_H
#define BOB_BLITZ_CONFIG_H

/* Define API version */
#define BOB_BLITZ_API_VERSION 0x0202


#ifdef BOB_IMPORT_VERSION

  /***************************************
  * Here we define some functions that should be used to build version dictionaries in the version.cpp file
  * There will be a compiler warning, when these functions are not used, so use them!
  ***************************************/

  #include <Python.h>
  #include <boost/preprocessor/stringize.hpp>


  /**
   * bob.blitz c/c++ api version
   */
  static PyObject* bob_blitz_version() {
    return Py_BuildValue("{ss}", "api", BOOST_PP_STRINGIZE(BOB_BLITZ_API_VERSION));
  }

#endif // BOB_IMPORT_VERSION

#endif // BOB_BLITZ_CONFIG_H
