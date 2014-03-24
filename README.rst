.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Thu 29 Aug 2013 16:07:57 CEST

.. image:: https://travis-ci.org/bioidiap/xbob.blitz.svg?branch=master
   :target: https://travis-ci.org/bioidiap/xbob.blitz
.. image:: https://coveralls.io/repos/bioidiap/xbob.blitz/badge.png
   :target: https://coveralls.io/r/bioidiap/xbob.blitz
.. image:: http://img.shields.io/github/tag/bioidiap/xbob.blitz.png
   :target: https://github.com/bioidiap/xbob.blitz
.. image:: http://img.shields.io/pypi/v/xbob.blitz.png
   :target: https://pypi.python.org/pypi/xbob.blitz
.. image:: http://img.shields.io/pypi/dm/xbob.blitz.png
   :target: https://pypi.python.org/pypi/xbob.blitz

====================================
 Python bindings for Blitz++ Arrays
====================================

This package contains a set of Pythonic bindings to the popular Blitz/C++
library. It also provides a C/C++ API that allows your Python extensions to
leverage from the interfaces provided by this package. For more information,
consult the package documentation.

Installation
------------

Install it through normal means, via PyPI or use ``zc.buildout`` to bootstrap
the package and run test units.

Documentation
-------------

You can generate the documentation for this package, after installation, using
Sphinx::

  $ sphinx-build -b html doc sphinx

This shall place in the directory ``sphinx``, the current version for the
documentation of the package.

Testing
-------

You can run a set of tests using the nose test runner::

  $ nosetests -sv xbob.blitz

You can run our documentation tests using sphinx itself::

  $ sphinx-build -b doctest doc sphinx

You can test overall test coverage with::

  $ nosetests --with-coverage --cover-package=xbob.blitz

The ``coverage`` egg must be installed for this to work properly.

Development
-----------

To develop this package, install using ``zc.buildout``, using the buildout
configuration found on the root of the package::

  $ python bootstrap.py
  ...
  $ ./bin/buildout

Tweak the options in ``buildout.cfg`` to disable/enable verbosity and debug
builds.
