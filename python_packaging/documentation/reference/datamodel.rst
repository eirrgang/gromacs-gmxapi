=================
gmxapi data model
=================

stub

Basic data types, containers
============================

Handles and Futures
===================

Proxies and managed resources
=============================

Operations, factories, and data flow: declaration, definition, and initialization
=================================================================================

Data proxies (Results, OutputCollections, PublishingDataProxies)
are created in a specific Context and maintain a reference
to the Context in which they are created.
The proxy is essentially a subscription to resources,
and the Context implementation will generally allow the
proxy to extend the life of resources that may be accessed
through the proxy. If the resources become unavailable,
access through the proxy must raise an informative exception
regarding the reason for the resource inaccessibility.


Examples could be

* the Context has finalized access to the resource, such as

  * after completing the definition of a subgraph
  * a write-once resource or iterator has already been used

Data
----

For compatibility and versatility, gmxapi data typing does not require specific
classes. In C++, typing uses C++ templating. In Python, abstract base classes
and duck typing are used. A C API provides a data description struct that is
easily convertible to the metadata structs for Python ctypes, numpy, Eigen, HDF5, etc.

Fundamental data types
~~~~~~~~~~~~~~~~~~~~~~

* Integer
* Float
* Boolean

Containers
~~~~~~~~~~

* NDArray
* String
* AssociativeArray

Constraints and placeholders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specify some parameters within a type.

Proxies
-------

* File()
* Future()
* Handle()

"""

Expressing inputs and outputs
-----------------------------

Notes on data compatibility
===========================

Avoid dependencies
------------------

The same C++ symbol can have different bindings in each extension module, so
don't rely on C++ typing through bindings. Need schema for PyCapsules.

Adding gmxapi compatible Python bindings should not require dependency on gmxapi
Python package. Compatibility through interfaces instead of inheritance.

