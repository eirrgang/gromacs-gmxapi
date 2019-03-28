# This file is part of the GROMACS molecular simulation package.
#
# Copyright (c) 2019, by the GROMACS development team, led by
# Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
# and including many others, as listed in the AUTHORS file in the
# top-level source directory and at http://www.gromacs.org.
#
# GROMACS is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# GROMACS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with GROMACS; if not, see
# http://www.gnu.org/licenses, or write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
#
# If you want to redistribute modifications to GROMACS, please
# consider that scientific software is very special. Version
# control is crucial - bugs must be traceable. We will be happy to
# consider code for inclusion in the official distribution, but
# derived work must not be called official GROMACS. Details are found
# in the README & COPYING files - if they are missing, get the
# official version at http://www.gromacs.org.
#
# To help us fund GROMACS development, we humbly ask that you cite
# the research papers on the package. Check out http://www.gromacs.org.

"""Define gmxapi-compliant Operations

Provide decorators and base classes to generate and validate gmxapi Operations.
"""

import abc
import collections
from contextlib import contextmanager
import functools
import inspect
import weakref

__all__ = ['computed_result',
           'append_list',
           'concatenate_lists',
           'function_wrapper',
           'make_constant',
           ]

from gmxapi import exceptions

# TODO: Instead of a base class to inherit, just provide a metaclass for validation.
#  abc.ABC is too strict for now and the base class does not provide any useful heritable functionality.
class AbstractResult(abc.ABC):
    """A Result serves as a proxy to or "future" for data produced by gmxapi operations.

    Result is the subset of the gmxapi Operation interface providing the output data proxy.

    An object implementing the Result interface has a `result()` method that forces resolution
    of data dependencies, triggering any necessary computation and communication, to return
    an object of the data type represented by the Result. If the Result is a container for other
    Results, nested Results are also resolved to local concrete data.

    Classes that provide named Results use a ResultDescriptor that provides a getter that returns
    an object with the Result interface.
    """
    # TODO: A Result should defer implementation details to the Context or parent operation.
    # Note: A Result instance should not need to hold more than a weakref, if even that, to a
    # Context instance, and no reference to an Operation instance.
    #@property
    # @abc.abstractmethod
    @property
    def uid(self):
        """Get a unique identifier by which this result can be identified.

        Value should be distinct for logically different input parameters (that would be expected
        to produce non-equivalent results.

        Value must be equal for any situation in which the Context should or must reuse resources
        to acquire the result.

        Value should be equal for logically equivalent inputs that would be expected to produce
        equivalent results (to allow duplication of work or data to be avoided).

        Value may be equal but is not required to be equal in different environments or for different
        output parameters, in which different artifacts may be produced. The use case would be to
        allow greater data portability without explicit conversion between contexts.

        Can be used by a Context implementation to determine whether data is available locally,
        manage checkpointing and caching, map inputs to outputs, etc.

        The `uid` property is likely to be a constant string that is set on creation of the Result object.
        """
        return False

    # TODO: (FR5+)
    # """An object implementing the Result interface (including a pure Result object) has
    # an `output` attribute that gets a pure Result copy of the Result handle.
    # """

    # @property
    # @abc.abstractmethod
    # def output(self):
    #     return self.resulttype(self)

    # TODO: A Result needs to provide a way to get/validate a handle in the (new) Context of a subscriber.

    # TODO: We might reformulate Result as derived from concurrent.futures.Future
    # but it is currently conceptually dissimilar in that the availability of a Result
    # does not depend on having `submit`ed a task, but rather on describing tasks to be submitted.
    @abc.abstractmethod
    def result(self):
        """Get the represented data as a local object."""
        # Maybe a generic implementation would look like this.
        # TODO: Check for valid resource manager.
        # TODO: Check whether uid is registered with resource manager
        # TODO: do we need to provide a reference to this object and/or its class
        #       to allow Context to lazily create and perform the operation at call time?
        self._resource_manager.get(self.uid)

    @property
    @abc.abstractmethod
    def dtype(self) -> type:
        """Base data type of the result.

        Used to determine compatibility with the mapped inputs of consuming operations.
        """
        # At any point in time, the resource represented by this result may be in some abstract state
        # and may by an array of data sources that will be scattered to consumers.
        return type(None)

    # def _subscribe(self, subscriber):
    #     """Register interest in being provided with the result data when it is available.
    #
    #     I don't think this is quite what we want. We want the consumer to register its interest
    #     through the resource management of the subscriber's Context, which can negotiate with
    #     the context of this object, probably by getting a Director from it.
    #     """
    #     pass
    #
    # @classmethod
    # def _director(self, input, context):
    #     """Get an operation director appropriate for the input and context."""
    #     director = None
    #     return director

# class ValidateResult(type):
#     """Metaclass for validating the Result interface.
#
#     Alternative to inheriting from AbstractResult.
#     """
#     def __new__(meta, name, bases, class_dict):
#         if not 'result' in class_dict or not callable(class_dict['result']):
#             message = "{} does not provide a callable 'result' attribute, required by the Result interface.".format(name)
#             raise exceptions.ApiError(message)
#         if not 'dtype' in class_dict or not isinstance(class_dict['dtype'], type):
#             message = "{} does not provide a 'dtype' attribute, required by the Result interface.".format(name)
#             raise exceptions.ApiError(message)
#         cls = type.__new__(meta, name, bases, class_dict)
#         return cls

# Result scenarios:
# In (rough) order of increasing complexity:
# * stateless and reproducible locally: calculate when needed
# * stateful and reproducible locally: calculate as needed, but implementation needs to avoid
#   resource contention, race conditions, reentrancy issues.
# * deferred: need to allow resource manager to provide data as it becomes available.
# In the general case, then, the Result handle should
# 1. allow a consumer to register its interest in the result with its own resource manager
#    and allow itself to be provided with the result when it is available.
# 2. Allow the holder of the Result handle to request the data immediately, with the understanding
#    that the surrounding code is blocked on the request.
# Note that in case (1), the holder of the handle may not use the facility, especially if it will
# be using (2).
# TODO: (FR5+) This class can be removed for tidiness when more sophisticated classes are avialable.
# E.g. caching Results, ensemble-safe results.
class ImmediateResult(AbstractResult):
    """Simple Result obtainable with local computation.

    Operation and result are stateless and can be evaluated in any Context.
    """

    def __init__(self, implementation, input):
        """`implementation` is idempotent and may be called repeatedly without (additional) side effects."""
        # Retain input information for introspection.
        assert callable(implementation)
        assert hasattr(input, 'args')
        assert hasattr(input, 'kwargs')
        self.__input = input
        self.__cached_value = implementation(*input.args, **input.kwargs)
        # TODO: (FR4) need a utility to resolve the base type of a value that may be a proxy object.
        self._dtype = type(self.__cached_value)

    @property
    def dtype(self):
        return self._dtype

    def result(self):
        return self.__cached_value


def computed_result(function):
    """Decorate a function to get a helper that produces an object with Result behavior.

    When called, the new function produces an ImmediateResult object.

    The new function has the same signature as the original function, but can accept
    proxy objects (Result objects) for arguments if the provided proxy objects represent
    a type compatible with the original signature.

    The ImmediateResult object will be evaluated in the local Context by the time its `result()`
    method returns the first time. Other than that, the API does not specify when input
    data dependencies will be resolved or when the wrapped function will be executed.

    Calls to `result()` return the value that `function` would return when executed in
    the local context with the inputs fully resolved.
    """
    @functools.wraps(function)
    def new_function(*args, **kwargs):
        # The signature of the new function will accept abstractions of whatever types it originally accepted.
        # * Create a mapping to the original call signature from `input`
        # * Add handling for typed abstractions in wrapper function.
        # * Process arguments to the wrapper function into `input`
        sig = inspect.signature(function)
        # Note: Introspection could fail.
        # TODO: Figure out what to do with exceptions where this introspection and rebinding won't work.
        # ref: https://docs.python.org/3/library/inspect.html#introspecting-callables-with-the-signature-object

        # TODO: (FR3+) create a serializable data structure for inputs discovered from function introspection.

        # TODO: (FR4) handle typed abstractions in input arguments

        input_list = []
        for arg in args:
            if hasattr(arg, 'result'):
                input_list.append(arg.result())
            else:
                input_list.append(arg)
        input_dict = {}
        for name, value in kwargs.items():
            if hasattr(value, 'result'):
                input_dict[name] = value.result()
            else:
                input_dict[name] = value

        input_pack = sig.bind(*input_list, **input_dict)

        result_object = ImmediateResult(function, input_pack)
        return result_object

    return new_function

@computed_result
def append_list(a:list=(), b:list=()):
    """Operation that consumes two lists and produces a concatenated single list."""
    # TODO: (FR3) Each sublist or sublist element could be a "future" handle; make sure input provider resolves that.
    # TODO: (FR4) Returned list should be an NDArray.
    for arg in (a, b):
        if isinstance(arg, (str, bytes)):
            raise exceptions.ValueError('Input must be a pair of lists.')
    try:
        list_a = list(a)
    except TypeError:
        list_a = list([a])
    try:
        list_b = list(b)
    except TypeError:
        list_b = list([b])
    return list_a + list_b

def concatenate_lists(sublists:list=()):
    """Trivial data flow restructuring operation to combine data sources into a single list."""
    if isinstance(sublists, (str, bytes)):
        raise exceptions.ValueError('Input must be a list of lists.')
    if len(sublists) == 0:
        return []
    else:
        return append_list(sublists[0], concatenate_lists(sublists[1:]))

@computed_result
def make_constant(value):
    """Create a source of the provided value.

    Accepts a value of any type. The object returned has a definite type.
    """
    return type(value)(value)

# In the longer term, Contexts could provide metaclasses that allow transformation or dispatching
# of the basic aspects of the operation protocols between Contexts or from a result handle into a
# new context, based on some attribute or behavior in the result handle.

# TODO: For outputs, distinguish between "results" and "events".
#  Both are published to the resource manager in the same way, but the relationship
#  with subscribers is potentially different.
def function_wrapper(output=None):
    """Generate a decorator for wrapped functions with signature manipulation.

    New function accepts the same arguments, with additional arguments required by
    the API.

    The new function returns an object with an `output` attribute containing the named outputs.

    Example:
        @function_wrapper(output={'spam': str, 'foo': str})
        def myfunc(parameter=None, output=None):
            output.spam = parameter
            output.foo = parameter + ' ' + parameter

        operation1 = myfunc(parameter='spam spam')
        assert operation1.output.spam.result() == 'spam spam'
        assert operation1.output.foo.result() == 'spam spam spam spam'
    """
    # TODO: more flexibility to capture return value by default?
    #     If 'output' is provided to the wrapper, a data structure will be passed to
    #     the wrapped functions with the named attributes so that the function can easily
    #     publish multiple named results. Otherwise, the `output` of the generated operation
    #     will just capture the return value of the wrapped function.
    # For now, this behavior is obtained with @computed_result

    # TODO: (FR5+) gmxapi operations need to allow a context-dependent way to generate an implementation with input.
    # This function wrapper reproduces the wrapped function's kwargs, but does not allow chaining a
    # dynamic `input` kwarg and does not dispatch according to a `context` kwarg. We should allow
    # a default implementation and registration of alternate implementations. We don't have to do that
    # with functools.singledispatch, but we could, if we add yet another layer to generate a wrapper
    # that takes the context as the first argument. (`singledispatch` inspects the first argument rather
    # that a named argument)

    # Implementation note: The closure of the current function is used to
    # dynamically define several classes that support the operation to be
    # created by the returned decorator.

    # Encapsulate the description of the input data flow.
    PyFuncInput = collections.namedtuple('Input', ('args', 'kwargs', 'dependencies'))

    # Encapsulate the description of a data output.
    Output = collections.namedtuple('Output', ('name', 'dtype', 'done', 'data'))

    class Publisher(object):
        """Data descriptor for write access to a specific named data resource.
        """
        def __init__(self, name, dtype):
            # self._input = Input(input.args, input.kwargs, input.dependencies)
            # self._instance = instance
            self.name = name
            self.dtype = dtype

        def __get__(self, instance, owner):
            if instance is None:
                # Access through class attribute of owner class
                return self
            resource_manager = instance._instance
            return getattr(resource_manager._data, self.name)

        def __set__(self, instance, value):
            resource_manager = instance._instance
            resource_manager.set_result(self.name, value)

    class DataProxyBase(object):
        """Limited interface to managed resources.

        Inherit from DataProxy to specialize an interface to an ``instance``.
        In the derived class, either do not define ``__init__`` or be sure to
        initialize the super class (DataProxy) with an instance of the object
        to be proxied.

        Acts as an owning handle to ``instance``, preventing the reference count
        of ``instance`` from going to zero for the lifetime of the proxy object.
        """
        def __init__(self, instance):
            self._instance = instance

    # Dynamically define a type for the PublishingDataProxy using a descriptor for each attribute.
    # TODO: Encapsulate this bit of script in a metaclass definition.
    namespace = {}
    for name, dtype in output.items():
        namespace[name] = Publisher(name, dtype)
    namespace['__doc__'] = "Handler for write access to the `output` of an operation.\n\n" + \
                           "Acts as a sort of PublisherCollection."
    PublishingDataProxy = type('PublishingDataProxy', (DataProxyBase,), namespace)

    class ResultGetter(object):
        """Fetch data to the caller's Context.

                    Returns an object of the concrete type specified according to
                    the operation that produces this Result.
                    """
        def __init__(self, resource_manager, name, dtype):
            self.resource_manager = resource_manager
            self.name = name
            self.dtype = dtype

        def __call__(self):
            # TODO: Tell the resource manager to resolve data dependencies
            self.resource_manager.update_output()
            assert self.resource_manager._data[self.name].done
            # Return ownership of concrete data
            return self.resource_manager._data[self.name].data

    class Future(object):
        def __init__(self, resource_manager, name, dtype):
            self.name = name
            if not isinstance(dtype, type):
                raise exceptions.ValueError('dtype argument must specify a type.')
            self.dtype = dtype
            # This abstraction anticipates that a Future might not retain a strong
            # reference to the resource_manager, but only to a facility that can resolve
            # the result() call. Additional aspects of the Future interface can be
            # developed without coupling to a specific concept of the resource manager.
            self._result = ResultGetter(resource_manager, name, dtype)

        def result(self):
            return self._result()

        def __getitem__(self, item):
            """Get a more limited view on the Future."""
            # TODO: Strict definition of outputs and output types can let us validate this earlier.
            #  We need AssociativeArray and NDArray so that we can type the elements. Allowing a
            #  Future with None type is a hack.
            result = lambda future=self, item=item : future.result()[item]
            future = collections.namedtuple('Future', ('dtype', 'result'))(None, result)
            return future

    class OutputDescriptor(object):
        """Read-only data descriptor for proxied output access.

        Knows how to get a Future from the resource manager.
        """
        def __init__(self, name, dtype):
            self.name = name
            self.dtype = dtype

        def __get__(self, proxy, owner):
            if proxy is None:
                # Access through class attribute of owner class
                return self
            return proxy._instance.future(name=self.name, dtype=self.dtype)

    class OutputDataProxy(DataProxyBase):
        """Handler for read access to the `output` member of an operation handle.

        Acts as a sort of ResultCollection.

        A ResourceManager creates an OutputDataProxy instance at initialization to
        provide the ``output`` property of an operation handle.
        """
        # TODO: Needs to know the output schema of the operation,
        #  so type definition is a detail of the operation definition. (Could be "templated" on Context type)
        # TODO: (FR3+) We probably want some other container behavior, in addition to the attributes...

    for name, dtype in output.items():
        setattr(OutputDataProxy, name, OutputDescriptor(name, dtype))

    class ResourceManager(object):
        """Provides data publication and subscription services.

        Owns data published by operation implementation or served to consumers.
        Mediates read and write access to the managed data streams.

        The `publisher` attribute is an object that can have pre-declared resources
        assigned to as attributes by an operation publishing its results.

        The `data` attribute is an object with pre-declared resources available through
        attribute access. Reading an attribute produces a new Result proxy object for
        pre-declared data or raises an AttributeError.

        TODO: This functionality should evolve to be a facet of Context implementations.
         There should be no more than one ResourceManager instance per work graph node in a Context.
         This will soon be at odds with letting the ResourceManager be owned by an operation instance handle.
        TODO: The publisher and data objects can be more strongly defined through interaction between the Context and clients.
        """

        @contextmanager
        def __publishing_context(self):
            """Get a context manager for resolving the data dependencies of this node.

            Use the returned object as a Python context manager.
            'output' type resources can be published exactly once, and only while the
            publishing context is active.

            Responsibilities of the context manager are to:
                * Make sure dependencies are resolved.
                * Prepare read-only part of resources (input)
                * Make sure outputs are marked 'done' before releasing lock.

            """

            # TODO:
            # if self._data.done():
            #     raise exceptions.ProtocolError('Resources have already been published.')
            resource = PublishingDataProxy(weakref.proxy(self))
            try:
                yield resource
            finally:
                self.done = True

        def __init__(self, input_fingerprint=None, runner=None):
            """Initialize a resource manager for the inputs and outputs of an operation.

            Arguments:
                runner : callable to be called once to set output data
                input_fingerprint : Uniquely identifiable input data description

            """
            assert callable(runner)
            assert input_fingerprint is not None

            # Note: This implementation assumes there is one ResourceManager instance per data source,
            # so we only stash the inputs and dependency information for a single set of resources.
            # TODO: validate input_fingerprint as its interface becomes clear.
            self._input_fingerprint = input_fingerprint
            self.__cached_input = None

            self._data = {name: Output(name=name, dtype=dtype, done=False, data=None) for name, dtype in output.items()}

            # TODO: reimplement as a data descriptor so that Publisher does not need a bound circular reference.
            self._publisher = PublishingDataProxy(weakref.proxy(self))
            self.__publishing_resources = [self.__publishing_context()]

            self.done = False
            self._runner = runner
            self.__operation_entrance_counter = 0

        def set_result(self, name, value):
            if type(value) == list:
                for item in value:
                    assert not hasattr(item, 'result')
            self._data[name] = Output(name=name,
                                      dtype=self._data[name].dtype,
                                      done=True,
                                      data=self._data[name].dtype(value))

        def update_output(self):
            """Bring the output of the bound operation up to date.

            Execute the bound operation once if and only if it has not
            yet been run in the lifetime of this resource manager.

            Used internally to implement Futures for the local operation
            associated with this resource manager.

            TODO: We need a different implementation for an operation whose output
             is served by multiple resource managers. E.g. an operation whose output
             is available across the ensemble, but which should only be executed on
             a single ensemble member.
            """
            # This code is not intended to be reentrant. We make a modest attempt to
            # catch unexpected reentrance, but this is not (yet) intended to be a thread-safe
            # resource manager implementation.
            if not self.done:
                self.__operation_entrance_counter += 1
                if self.__operation_entrance_counter > 1:
                    raise exceptions.ProtocolError('Bug detected: resource manager tried to execute operation twice.')
                if not self.done:
                    with self.local_input() as input:
                        # Note: Resources are marked "done" by the resource manager
                        # when the following context manager completes.
                        # TODO: Allow both structured and singular output.
                        #  For simple functions, just capture and publish the return value.
                        with self.publishing_resources() as output:
                            self._runner(*input.args, output=output, **input.kwargs)

        def future(self, name:str=None, dtype=None):
            """Retrieve a Future for a named output.

            TODO: (FR5+) Normalize this part of the interface between operation definitions and
             resource managers.
            """
            if not isinstance(name, str) or not name in self._data:
                raise exceptions.ValueError('"name" argument must name an output.')
            assert dtype is not None
            if dtype != self._data[name].dtype:
                message = 'Requested Future of type {} is not compatible with available type {}.'
                message = message.format(dtype, self._data[name].dtype)
                raise exceptions.ApiError(message)
            return Future(self, name, dtype)

        def data(self):
            """Get an adapter to the output resources to access results."""
            return OutputDataProxy(self)

        @contextmanager
        def local_input(self):
            """In an API session, get a handle to fully resolved locally available input data.

            Execution dependencies are resolved on creation of the context manager. Input data
            becomes available in the ``as`` object when entering the context manager, which
            becomes invalid after exiting the context manager. Resources allocated to hold the
            input data may be released when exiting the context manager.

            It is left as an implementation detail whether the context manager is reusable and
            under what circumstances one may be obtained.
            """
            # Localize data
            # TODO: (FR3) take action only if outputs are not already done.
            # TODO: (FR3) make sure this gets run if outputs need to be satisfied for `result()`
            for dependency in self._dependencies:
                dependency()

            # TODO: (FR3+) be more rigorous.
            #  This should probably also use a sort of Context-based pub-sub rather than
            #  the result() method, which is explicitly for moving data across the API boundary.
            args = []
            try:
                for arg in self._input_fingerprint.args:
                    if hasattr(arg, 'result'):
                        args.append(arg.result())
                    else:
                        args.append(arg)
            except Exception as E:
                raise exceptions.ApiError('input_fingerprint not iterating on "args" attr as expected.') from E

            kwargs = {}
            try:
                for key, value in self._input_fingerprint.kwargs.items():
                    if hasattr(value, 'result'):
                        kwargs[key] = value.result()
                    else:
                        kwargs[key] = value
                    if isinstance(kwargs[key], list):
                        new_list = []
                        for item in kwargs[key]:
                            if hasattr(item, 'result'):
                                new_list.append(item.result())
                            else:
                                new_list.append(item)
                        kwargs[key] = new_list
                    try:
                        for item in kwargs[key]:
                            # TODO: This should not happen. Need proper tools for NDArray Futures.
                            # assert not hasattr(item, 'result')
                            if hasattr(item, 'result'):
                                kwargs[key][item] = item.result()
                    except TypeError:
                        # This is only a test for iterables
                        pass
            except Exception as E:
                raise exceptions.ApiError('input_fingerprint not iterating on "kwargs" attr as expected.') from E

            assert 'input' not in kwargs

            for key, value in kwargs.items():
                if key == 'command':
                    if type(value) == list:
                        for item in value:
                            assert not hasattr(item, 'result')
            input_pack = collections.namedtuple('InputPack', ('args', 'kwargs'))(args, kwargs)

            # Prepare input data structure
            yield input_pack

        def publishing_resources(self):
            """Get a context manager for resolving the data dependencies of this node.

            Use the returned object as a Python context manager.
            'output' type resources can be published exactly once, and only while the
            publishing context is active.

            Write access to publishing resources can be granted exactly once during the
            resource manager lifetime and conveys exclusive access.
            """
            return self.__publishing_resources.pop()

        ###
        # TODO: Need a facility to resolve inputs, chasing dependencies...
        ###

        @property
        def _dependencies(self):
            """Generate a sequence of call-backs that notify of the need to satisfy dependencies."""
            for arg in self._input_fingerprint.args:
                if hasattr(arg, 'result') and callable(arg.result):
                    yield arg.result
            for _, arg in self._input_fingerprint.kwargs.items():
                if hasattr(arg, 'result') and callable(arg.result):
                    yield arg.result
            for item in self._input_fingerprint.dependencies:
                assert hasattr(item, 'run')
                yield item.run


    def decorator(function):
        @functools.wraps(function)
        def factory(**kwargs):

            def get_resource_manager(instance):
                """Provide a reference to a resource manager for the dynamically defined Operation.

                Initial Operation implementation must own ResourceManager. As more formal Context is
                developed, this can be changed to a weak reference. A distinction can also be developed
                between the facet of the Context-level resource manager to which the Operation has access
                and the whole of the managed resources.
                """
                return ResourceManager(input_fingerprint=instance._input, runner=function)

            class Operation(object):
                """Dynamically defined Operation implementation.

                Define a gmxapi Operation for the functionality being wrapped by the enclosing code.
                """
                signature = inspect.signature(function)

                def __init__(self, **kwargs):
                    """Initialization defines the unique input requirements of a work graph node.

                    Initialization parameters map to the parameters of the wrapped function with
                    addition(s) to support gmxapi data flow and deferred execution.

                    If provided, an ``input`` keyword argument is interpreted as a parameter pack
                    of base input. Inputs also present as standalone keyword arguments override
                    values in ``input``.

                    Inputs that are handles to gmxapi operations or outputs induce data flow
                    dependencies that the framework promises to satisfy before the Operation
                    executes and produces output.
                    """
                    ## Define the unique identity and data flow constraints of this work graph node.
                    # TODO: (FR3) generalize
                    input_dependencies = []

                    # TODO: Resolve the rigor of the input schema.
                    # 1. Input structure is strongly specified / allowed parameters defined in class.
                    #    The class definition does not need to be nested in new_helper. Inspection of
                    #    the class makes it easy for the framework to preprocess arguments and provide
                    #    localized data.
                    # 2. Input is only specified as far as being a Map.
                    #    Contents of the Map need to be processed (recursively TBD) to resolve all data flow.
                    #
                    # TODO: Make allowed input strongly specified in the Operation definition.
                    # TODO: Resolve execution dependencies at run() and make non-data
                    #  execution `dependencies` just another input that takes the default
                    #  output of an operation and doesn't do anything with it.

                    # If present, kwargs['input'] is treated as an input "pack" providing _default_ values.
                    input_kwargs = {}
                    if 'input' in kwargs:
                        provided_input = kwargs.pop('input')
                        if provided_input is not None:
                            # Try to determine what 'input' is.
                            # TODO: (FR5+) handling should be related to Context.
                            #  The process of accepting input arguments includes resolving placement in
                            #  a work graph and resolving the Context responsibilities for graph nodes.
                            if hasattr(provided_input, 'run'):
                                input_dependencies.append(provided_input)
                            else:
                                # Assume a parameter pack is provided.
                                for key, value in provided_input.items():
                                    input_kwargs[key] = value
                    assert 'input' not in kwargs
                    assert 'input' not in input_kwargs

                    # Merge kwargs and kwargs['input'] (keyword parameters versus parameter pack)
                    for key in kwargs:
                        if key in self.signature.parameters:
                            input_kwargs[key] = kwargs[key]
                        else:
                            raise exceptions.UsageError('Unexpected keyword argument: {}'.format(key))

                    # TODO: Check input types

                    self.__input = PyFuncInput(args=[],
                                         kwargs=input_kwargs,
                                         dependencies=input_dependencies)
                    ##

                    # TODO: (FR5+) Split the definition of the resource structure and the resource initialization.
                    # Resource structure definition logic can be moved to the level of the class definition.
                    # We need knowledge of the inputs to uniquely identify the resources for this operation instance.
                    # Implementation suggestion: Context-provided metaclass defines resource manager
                    # interface for this Operation. Factory function initializes compartmentalized
                    # resource management at object creation.
                    self.__resource_manager = get_resource_manager(self)

                @property
                def _input(self):
                    """Internal interface to support data flow and execution management."""
                    return self.__input

                @property
                def output(self):
                    # Note: if we define Operation classes exclusively in the scope of Context instances,
                    # we could elegantly have a single _resource_manager handle instance per Operation type
                    # per Context instance. That could make it easier to implement library-level optimizations
                    # for managing hardware resources or data placement for operations implemented in the
                    # same librarary. That would be well in the future, though, and could also be accomplished
                    # with other means, so here I'm assuming one resource manager handle instance per Operation handle instance.
                    #
                    # TODO: Allow both structured and singular output.
                    #  Either return self._resource_manager.data or self._resource_manager.data.output
                    # TODO: We can configure `output` as a data descriptor instead of a property so that we
                    #  can get more information from the class attribute before creating an instance.
                    # The C++ equivalence would probably be a templated free function for examining traits.
                    return self.__resource_manager.data()

                # TODO: (FR5+) This behavior should be composed in with help from the Context implementation.
                def run(self):
                    """Make a single attempt to resolve data flow conditions.

                    This is a public method, but should not need to be called by users. Instead,
                    just use the `output` data proxy for result handles, or force data flow to be
                    resolved with the `result` methods on the result handles.

                    `run()` may be useful to try to trigger computation (such as for remotely
                    dispatched work) without retrieving results locally right away.

                    `run()` is also useful internally as a facade to the Context implementation details
                    that allow `result()` calls to ask for their data dependencies to be resolved.
                    Typically, `run()` will cause results to be published to subscribing operations as
                    they are calculated, so the `run()` hook allows execution dependency to be slightly
                    decoupled from data dependency, as well as to allow some optimizations or to allow
                    data flow to be resolved opportunistically. `result()` should not call `run()`
                    directly, but should cause the resource manager / Context implementation to process
                    the data flow graph.

                    In one conception, `run()` can have a return value that supports control flow
                    by itself being either runnable or not. The idea would be to support
                    fault tolerance, implementations that require multiple iterations / triggers
                    to complete, or looping operations.
                    """
                    self.__resource_manager.update_output()

            operation = Operation(**kwargs)
            return operation

        return factory

    return decorator


def generic_function(function):
    """Create a prototype for an operation with incompletely specified inputs and outputs.

    Extend function_wrapper to allow modification to the generated factory.

    Wrap a generic function to produce a partially specified Operation. Product can be
    converted to a fully specified operation by

    A fully-specified Operation has all inputs named and typed so that data flow constraints
    are clear.

    Create new fused operations by adding output methods to the factory.

    Note that the factory produces a fundamentally different Operation if modified between
    uses.

    TODO: Consider merging with function_wrapper() or otherwise restructuring.
    """
