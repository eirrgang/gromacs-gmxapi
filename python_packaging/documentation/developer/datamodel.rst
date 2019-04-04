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


Expressing inputs and outputs
-----------------------------

An operation type must express its allowed inputs (in order to be able to bind
at initialization of new instances).

An operation instance must express well-defined available outputs. Note that an
"instance" may not be runnable in all contexts, but must be inspectable such that
the context of operationB can inspect the outputs of operationA to determine
compatibility.

Future types versus Proxy types
-------------------------------

Future types require explicit action to convert to directly-accessible data via
the `result()` call, whether or not data flow resolution is necessary. Data is
not writeable through the Future handle.

Proxy types resolve data dependencies as necessary when converted to native types.
(In Python, they express `__int__`, `__float__`, etc.)
Proxy types may be writeable, but are obtained with access controls.

Consider `memoryview` as a model for proxies and Results: has a `release()`
method that is called automatically when handle is obtained in a context manager,
after which accesses produce
`ValueError: operation forbidden on released memoryview object`

Arrays
======

Version 1: NDArray handle is opaque: not iterable
Version 2: ensemble and NDArray follow strict hierarchy of dimensional rank
Version 3: ensemble and NDArray merged with numpy-like dimensionality and rank transformations?

Version 1 Ensemble input determination
--------------------------------------

Scalar input

function fill_from_scalar_source(input, source):
    try:
        input.set(input.dtype(source))
    else try:
        input.set(gmxapi_future(source, dtype=input.dtype))
    else try:
        if iterable(source) and not isinstance(source, (str, bytes)):
            for i, element in enumerate(source):
                input.ensemble_rank(i).set(fill_from_scalar_source(input.ensemble_rank(i), element))

Array input treated as a type of scalar

Version 2 Ensemble input determination
--------------------------------------

Scalar input: get handle to dimension 0

function fill_from_scalar_source(input, source):
    try:
        input[...] = input.dtype(source)
    else try:
        input[...] = gmxapi_future(source)
    else try:
        if iterable(source) and not isinstance(source, (str, bytes)):
            for i, element in enumerate(source):
                fill_from_scalar(input[i], element)

(specify recursion depth.)
Generators must be explicitly resolved or converted to futures for v1.

array input

function get_array_input(source, N):
    if isinstance(source, (str, bytes):
        if issubclass(input.dtype, (str, bytes)):
            fill_from_scalar(input[...], source)

    try:
        # could broadcast up or down
        input(N) = gmxapi_future(source)
    else:
        # could broadcast up or down
        input(N) = from_buffer(source)
    else:
        if iterable(source):
            foreach element in source:
                input(N-1)[:] = get_array_input(element)
            input(N-1) = get_array_input(

Input tries to consume the source as a _gmxapi_future.
Input tries to consume source as a non-str, non-bytes buffer.
Input tries to consume source as a memoryview-like object.
Input tries to consume source as a sequence of compatible Scalar input (see above)

map input

Input tries to consume the source as a _gmxapi_future.


1. Can the input consume the source?
    a. Scalar -> Scalar: yes. not ensemble
    b.

Input argument is assumed to be an ensemble of values if it

1. does not implement the _gmxapi_future interface
2. is iterable
3. not isinstace(arg, (str, bytes)
4a. is not a generator and has dimensionality that is greater than the consuming input
4b. is a generator or has dimensionality greater than the consuming input
4c.

Note: This implies that numpy.ndarray requires explicit wrapping to avoid being
considered as ensemble input.

Operation implementation
------------------------

The implementation expresses its named inputs and their types. The framework
guarantees that the operation will be provided with input of the indicated type
and structure when called.

The framework considers input compatible if the input is a compatible type or
future of a compatible type, or if the input is an ensemble of compatible input.

In the Python implementation, the framework checks the expressed input type and
resolves the abstract base class / metaclass. To type-check input arguments, the
framework can perform the following checks.
1. If the input object has a `_gmxapi_future` attribute, the Data Future Protocol
   is used to confirm compatibility and bind. All gmxapi types can implement the
   Data Future Protocol.
2. If the input is Iterable and not a string or bytes

Note: need to warn users that `bytes` will be interpreted as utf-8 encoded strings,
and that if they want to provide binary data through the Python buffer interface,
they should not do so by subclassing `bytes`, or they should first wrap their `bytes`
derived object with `memoryview()` or `gmxapi.ndarray()`



Data Future protocol
--------------------


# Result scenarios:
#
# In (rough) order of increasing complexity:
#
# * stateless and reproducible locally: calculate when needed
# * stateful and reproducible locally: calculate as needed, but implementation
#   needs to avoid resource contention, race conditions, reentrancy issues.
# * deferred: need to allow resource manager to provide data as it becomes available.
#
# In the general case, then, the Result handle should
#
# 1. allow a consumer to register its interest in the result with its own resource
#    manager and allow itself to be provided with the result when it is available.
# 2. Allow the holder of the Result handle to request the data immediately,
#    with the understanding that the surrounding code is blocked on the request.
#
# Note that in case (1), the holder of the handle may not use the facility,
# especially if it will be using (2).


# Questions:
#  * Are the members of `output` statically specified?
#  * Are the keys of a Map statically specified?
#  * Is `output` a Map?
# Answers:
# Compiled code should be able to discover an output format. A Map may have different keys depending
# on the work and user input, even when consumed or produced by compiled code. (A Map with statically
# specified keys would be a schema, which will not be implemented for a while.) Therefore, `output`
# is not a Map or a Result of Map type, but a ResultCollection or ResultCollectionDescriptor
# (which may be the output version of the future schema implementation).


Notes on data compatibility
===========================

Avoid dependencies
------------------

The same C++ symbol can have different bindings in each extension module, so
don't rely on C++ typing through bindings. Need schema for PyCapsules.

Adding gmxapi compatible Python bindings should not require dependency on gmxapi
Python package. Compatibility through interfaces instead of inheritance.

