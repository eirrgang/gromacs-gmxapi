===========================
gmx Python module reference
===========================

.. contents:: :local:
    :depth: 2

.. Concise reference documentation extracted directly from code.
.. For new and non-backwards-compatible features, API versions must be given.

The Gromacs Python interface is implemented as a high-level scripting interface implemented in pure Python and a
lower-level API implemented as a C++ extension.
The pure Python implementation provides the basic ``gmxampi`` module and
classes with a stable syntax that can be maintained with maximal compatibility
while mapping to lower level interfaces that may take a while to sort out. The
separation also serves as a reminder that different execution contexts may be
implemented quite diffently, though Python scripts using only the high-level
interface should execute on all. Bindings to the ``libgromacs`` C++ API are
provided in the submodule :mod:`gmxapi._gmxapi`.

Refer to the Python source code itself for additional clarification.

.. Configuration for doctest: automated syntax checking in documentation code snippets
.. testsetup::

    import gmxapi as gmx

.. _python-procedural:

User reference
==============

.. alphabetize the functions except for gmx.run (put first)

.. py:function:: read_tpr()

    Get a handle to a TPR run input file.

Exceptions
==========

.. automodule:: gmxapi.exceptions
   :members:

Python API
==========

.. contents:: :local:

