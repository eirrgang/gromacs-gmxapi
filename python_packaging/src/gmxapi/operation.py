#
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

Nodes in a work graph are created as instances of Operations. An Operation factory
accepts well-defined inputs as key word arguments. The object returned by such
a factory is a handle to the node in the work graph. It's ``output`` attribute
is a collection of the Operation's results.

function_wrapper(...) produces a wrapper that converts a function to an Operation
factory. The Operation is defined when the wrapper is called. The Operation is
instantiated when the factory is called. The function is executed when the Operation
instance is run.

The framework ensures that an Operation instance is executed no more than once.
"""

__all__ = ['computed_result',
           'join_arrays',
           'concatenate_lists',
           'function_wrapper',
           'make_constant',
           ]

import collections
import functools
import inspect
import weakref
from contextlib import contextmanager
from typing import Sequence, TypeVar

import gmxapi as gmx
from gmxapi.datamodel import NDArray, InputCollection, ndarray
from gmxapi import exceptions


def computed_result(function):
    """Decorate a function to get a helper that produces an object with Result behavior.

    When called, the new function produces an ImmediateResult object.

    The new function has the same signature as the original function, but can accept
    gmxapi data proxies, assuming the provided proxy objects represent types
    compatible with the original signature.

    Calls to `result()` return the value that `function` would return when executed
    in the local context with the inputs fully resolved.

    The API does not specify when input data dependencies will be resolved
    or when the wrapped function will be executed. That is, ``@computed_result``
    functions may force immediate resolution of data dependencies and/or may
    be called more than once to satisfy dependent operation inputs.
    """
    try:
        sig = inspect.signature(function)
    except TypeError as T:
        raise exceptions.ApiError('Can not inspect type of provided function argument.') from T
    except ValueError as V:
        raise exceptions.ApiError('Can not inspect provided function signature.') from V
    # Note: Introspection could fail.
    # Note: ApiError indicates a bug because we should handle this more intelligently.
    # TODO: Figure out what to do with exceptions where this introspection
    #  and rebinding won't work.
    # ref: https://docs.python.org/3/library/inspect.html#introspecting-callables-with-the-signature-object

    # 1. Get name and valid gmxapi type for all arguments.
    # 2. Define the Input data structure for the dependency.
    # 3. Create an instance of the Input from the provided arguments.
    # 4. Bind data from the Input to the function to be called.
    input_collection = InputCollection(sig)

    @functools.wraps(function)
    def new_function(*args, **kwargs):
        # The signature of the new function will accept abstractions
        # of whatever types it originally accepted. This wrapper must
        # * Create a mapping to the original call signature from `input`
        # * Add handling for typed abstractions in wrapper function.
        # * Process arguments to the wrapper function into `input`

        # 1. Inspect the return annotation to determine valid gmxapi type(s)
        # 2. Generate a Result object advertising the correct type, bound to the
        #    Input and implementing function.
        # 3. Transform the result() data to the correct type.

        # TODO: (FR3+) create a serializable data structure for inputs discovered
        #  from function introspection.

        # TODO: (FR4) handle typed abstractions in input arguments

        for name, param in sig.parameters.items():
            assert not param.kind == param.POSITIONAL_ONLY
        bound_arguments = sig.bind(*args, **kwargs)
        wrapped_function = function_wrapper()(function)
        handle = wrapped_function(**bound_arguments.arguments)
        output = handle.output
        return output.data

    return new_function


@computed_result
def join_arrays(a: list = (), b: list = ()) -> list:
    """Operation that consumes two sequences and produces a concatenated single sequence.

    Note that the exact signature of the operation is not determined until this
    helper is called. Helper functions may dispatch to factories for different
    operations based on the inputs. In this case, the dtype and shape of the
    inputs determines dtype and shape of the output. An operation instance must
    have strongly typed output, but the input must be strongly typed on an
    object definition so that a Context can make runtime decisions about
    dispatching work and data before instantiating.
    # TODO: elaborate and clarify.
    # TODO: check type and shape.
    # TODO: figure out a better annotation.
    """
    # TODO: (FR4) Returned list should be an NDArray.
    if isinstance(a, (str, bytes)) or isinstance(b, (str, bytes)):
        raise exceptions.ValueError('Input must be a pair of lists.')
    list_a = []
    list_a.extend(a)
    list_a.extend(b)
    return list_a


Scalar = TypeVar('Scalar')


def concatenate_lists(sublists: list = ()) -> NDArray:
    """Combine data sources into a single list.

    A trivial data flow restructuring operation
    """
    if isinstance(sublists, (str, bytes)):
        raise exceptions.ValueError('Input must be a list of lists.')
    if len(sublists) == 0:
        return ndarray([])
    else:
        return join_arrays(a=sublists[0], b=concatenate_lists(sublists[1:]))


def make_constant(value: Scalar) -> Scalar:
    """Provide a predetermined value at run time.

    This is a trivial operation that provides a (typed) value, primarily for
    internally use to manage gmxapi data flow.

    Accepts a value of any type. The object returned has a definite type and
    provides same interface as other gmxapi outputs. Additional constraints or
    guarantees on data type may appear in future versions.
    """
    # TODO: (FR4+) Manage type compatibility with gmxapi data interfaces.
    scalar_type = type(value)
    assert not isinstance(scalar_type, type(None))
    operation = function_wrapper(output={'data': scalar_type})(lambda data=scalar_type(): data)
    return operation(data=value).output.data


class Publisher(object):
    """Data descriptor for write access to a specific named data resource.

    For a wrapped function receiving an ``output`` argument, provides the
    accessors for an attribute on the object passed as ``output``. Maps
    read and write access by the wrapped function to appropriate details of
    the resource manager.

    Used internally to implement settable attributes on PublishingDataProxy.
    Allows PublishingDataProxy to be dynamically defined in the scope of the
    operation.function_wrapper closure. Each named output is represented by
    an instance of Publisher in the PublishingDataProxy class definition for
    the operation.

    Ref: https://docs.python.org/3/reference/datamodel.html#implementing-descriptors

    Collaborations:
    Relies on implementation details of ResourceManager.
    """

    def __init__(self, name: str):
        # self._input = Input(input.args, input.kwargs, input.dependencies)
        # self._instance = instance
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            # Access through class attribute of owner class
            return self
        resource_manager = instance._instance
        return getattr(resource_manager._data, self.name)

    def __set__(self, instance, value):
        resource_manager = instance._instance
        resource_manager.set_result(self.name, value)

    def __repr__(self):
        return 'Publisher(name={}, dtype={})'.format(self.name, self.dtype.__qualname__)


class DataProxyBase(object):
    """Limited interface to managed resources.

    Inherit from DataProxy to specialize an interface to an ``instance``.
    In the derived class, either do not define ``__init__`` or be sure to
    initialize the super class (DataProxy) with an instance of the object
    to be proxied.

    Acts as an owning handle to ``instance``, preventing the reference count
    of ``instance`` from going to zero for the lifetime of the proxy object.
    """

    # This class can be expanded to be the attachment point for a metaclass for
    # data proxies such as PublishingDataProxy or OutputDataProxy, which may be
    # defined very dynamically and concisely as a set of Descriptors and a type()
    # call.
    # If development in this direction does not materialize, then this base
    # class is not very useful and should be removed.
    def __init__(self, instance):
        self._instance = instance


def define_publishing_data_proxy(output_description):
    """Returns a class definition for a PublishingDataProxy for the provided output description."""
    # Dynamically define a type for the PublishingDataProxy using a descriptor for each attribute.
    # TODO: Encapsulate this bit of script in a metaclass definition?
    namespace = {}
    # Note: uses `output` from outer closure
    for name, dtype in output_description.items():
        namespace[name] = Publisher(name)
    namespace['__doc__'] = "Handler for write access to the `output` of an operation.\n\n" + \
                           "Acts as a sort of PublisherCollection."
    return type('PublishingDataProxy', (DataProxyBase,), namespace)


class ResultGetter(object):
    """Fetch data to the caller's Context.

    Returns an object of the concrete type specified according to
    the operation that produces this Result.
    """

    def __init__(self, resource_manager, name, data_description: gmx.datamodel.ResultDescription):
        self.resource_manager = resource_manager
        self.name = name
        assert isinstance(data_description, gmx.datamodel.ResultDescription)
        self.data_description = data_description

    def __call__(self):
        self.resource_manager.update_output()
        assert self.resource_manager._data[self.name].done
        # Return ownership of concrete data
        handle = self.resource_manager._data[self.name]
        if handle._description.dtype == NDArray:
            return handle.data.values
        else:
            return handle.data


class Future(object):
    def __init__(self, resource_manager, name: str = '', description: gmx.datamodel.ResultDescription = None):
        self.name = name
        if not isinstance(description, gmx.datamodel.ResultDescription):
            raise exceptions.ValueError('Need description of requested data.')
        self.description = description
        # This abstraction anticipates that a Future might not retain a strong
        # reference to the resource_manager, but only to a facility that can resolve
        # the result() call. Additional aspects of the Future interface can be
        # developed without coupling to a specific concept of the resource manager.

        self._result = ResultGetter(resource_manager, name, description)

    def result(self):
        return self._result()

    @property
    def dtype(self):
        return self.description.dtype

    def __getitem__(self, item):
        """Get a more limited view on the Future."""

        # TODO: Strict definition of outputs and output types can let us validate this earlier.
        #  We need AssociativeArray and NDArray so that we can type the elements.
        #  Allowing a Future with None type is a hack.
        def result():
            return self.result()[item]

        future = collections.namedtuple('Future', ('dtype', 'result'))(None, result)
        return future


class OutputDescriptor(object):
    """Read-only data descriptor for proxied output access.

    Knows how to get a Future from the resource manager.
    """

    def __init__(self, name, dtype):
        self.name = name
        assert isinstance(dtype, type)
        assert issubclass(dtype, (str, bool, int, float, dict, NDArray))
        self.dtype = dtype

    def __get__(self, proxy, owner):
        if proxy is None:
            # Access through class attribute of owner class
            return self
        result_description = gmx.datamodel.ResultDescription(dtype=self.dtype, width=1)
        return proxy._instance.future(name=self.name, description=result_description)


def define_output_data_proxy(output_description: gmx.datamodel.OutputCollectionDescription):
    class OutputDataProxy(DataProxyBase):
        """Handler for read access to the `output` member of an operation handle.

        Acts as a sort of ResultCollection.

        A ResourceManager creates an OutputDataProxy instance at initialization to
        provide the ``output`` property of an operation handle.
        """
        # TODO: Needs to know the output schema of the operation,
        #  so type definition is a detail of the operation definition.
        #  (Could be "templated" on Context type)
        # TODO: (FR3+) We probably want some other container behavior,
        #  in addition to the attributes...

    # Note: the OutputDataProxy has an inherent ensemble shape in the context
    # in which it is used, but that is an instance characteristic, not part of this type definition.

    for name, description in output_description.items():
        # TODO: (FR5) The current tool does not support topology changing operations.
        setattr(OutputDataProxy, name, OutputDescriptor(name, description))

    return OutputDataProxy


# In the longer term, Contexts could provide metaclasses that allow transformation or dispatching
# of the basic aspects of the operation protocols between Contexts or from a result handle into a
# new context, based on some attribute or behavior in the result handle.


# Encapsulate the description of the input data flow.
PyFuncInput = collections.namedtuple('Input', ('args', 'kwargs', 'dependencies'))


# Encapsulate the description and storage of a data output.
class OutputData(object):
    def __init__(self, name: str = '', description: gmx.datamodel.ResultDescription = None):
        assert name != ''
        self._name = name
        assert isinstance(description, gmx.datamodel.ResultDescription)
        self._description = description
        self._done = False
        self._data = None

    @property
    def name(self):
        return self._name

    @property
    def done(self):
        return self._done

    @property
    def data(self):
        if not self.done:
            raise exceptions.ApiError('Attempt to read before data has been published.')
        if self._data is None:
            raise exceptions.ApiError('Data marked "done" but contains null value.')
        return self._data

    def set(self, value):
        if self._description.dtype == NDArray:
            self._data = gmx.datamodel.ndarray(value)
        else:
            self._data = self._description.dtype(value)
        self._done = True


class ResourceManager(object):
    """Provides data publication and subscription services.

        Owns the data published by the operation implementation or served to consumers.
        Mediates read and write access to the managed data streams.

        This ResourceManager implementation is defined in conjunction with a
        run-time definition of an Operation that wraps a Python callable (function).
        ResourceManager is instantiated with a reference to the callable.

        When the Operation is run, the resource manager prepares resources for the wrapped
        function. Inputs provided to the Operation factory are provided to the
        function as keyword arguments. The wrapped function publishes its output
        through the (additional) ``output`` key word argument. This argument is
        a short-lived resource, prepared by the ResourceManager, with writable
        attributes named in the call to function_wrapper().

        After the Operation has run and the outputs published, the data managed
        by the ResourceManager is marked "done."

        Protocols:

        The data() method produces a read-only collection of outputs named for
        the Operation when the Operation's ``output`` attribute is accessed.

        publishing_resources() can be called once during the ResourceManager lifetime
        to provide the ``output`` object for the wrapped function. (Used by update_output().)

        update_output() brings the managed output data up-to-date with the input
        when the Operation results are needed. If the Operation has not run, an
        execution session is prepared with input and output arguments for the
        wrapped Python callable. Output is publishable only during this session.

    TODO: This functionality should evolve to be a facet of Context implementations.
     There should be no more than one ResourceManager instance per work graph
     node in a Context. This will soon be at odds with letting the ResourceManager
     be owned by an operation instance handle.
    TODO: The publisher and data objects can be more strongly defined through
     interaction between the Context and clients.

    Design notes:

    The normative pattern for updating data is to execute a node in the work
    graph, passing Resources for an execution Session to an operation runner.
    The resources and runner are dependent on the implementation details of
    the operation and the execution context, so logical execution may look
    like the following.

        resource_builder = ResourcesBuilder()
        runner_builder = RunnerBuilder()
        input_resource_director = input_resource_factory.director(input)
        output_resource_director = publishing_resource_factory.director(output)
        input_resource_director(resource_builder, runner_builder)
        output_resource_director(resource_builder, runner_builder)
        resources = resource_builder.build()
        runner = runner_builder.build()
        runner(resources)

    Only the final line is intended to be literal. The preceding code, if it
    exists in entirety, may be spread across several code comments.
    """

    @contextmanager
    def __publishing_context(self):
        """Get a context manager for resolving the data dependencies of this node.

            The returned object is a Python context manager (used to open a `with` block)
            to define the scope in which the operation's output can be published.
            'output' type resources can be published exactly once, and only while the
            publishing context is active. (See operation.function_wrapper())

            Used internally to implement ResourceManager.publishing_resources()

            Responsibilities of the context manager are to:
                * (TODO) Make sure dependencies are resolved.
                * Make sure outputs are marked 'done' when leaving the context.

            """

        # TODO:
        # if self._data.done():
        #     raise exceptions.ProtocolError('Resources have already been published.')
        resource = self._operation.PublishingDataProxy(weakref.proxy(self))
        # ref: https://docs.python.org/3/library/contextlib.html#contextlib.contextmanager
        try:
            yield resource
        except Exception as e:
            message = 'Uncaught exception while providing output-publishing resources for {}.'.format(self._runner)
            raise exceptions.ApiError(message) from e
        finally:
            self.done = True

    def __init__(self, input_fingerprint=None, operation=None):
        """Initialize a resource manager for the inputs and outputs of an operation.

        Arguments:
            operation : implementation details for a Python callable
            input_fingerprint : Uniquely identifiable input data description

        """
        runner = operation.runner
        assert callable(runner)
        assert input_fingerprint is not None

        # Note: This implementation assumes there is one ResourceManager instance per data source,
        # so we only stash the inputs and dependency information for a single set of resources.
        # TODO: validate input_fingerprint as its interface becomes clear.
        self._input_fingerprint = input_fingerprint
        self._operation = operation

        self._data = self._operation.make_datastore()

        # TODO: reimplement as a data descriptor
        #  so that PublishingDataProxy does not need a bound circular reference.
        self.__publishing_resources = [self.__publishing_context]

        self.done = False
        self._runner = runner
        self.__operation_entrance_counter = 0

    def set_result(self, name, value):
        try:
            for item in value:
                # In this specification, it is antithetical to publish Futures.
                if hasattr(item, 'result'):
                    raise exceptions.ApiError('Operation produced Future instead of real output.')
        except TypeError:
            # Ignore when `item` is not iterable.
            pass
        # type_annotation = self._data[name].dtype
        # type_caster = self._data[name].dtype
        # if type_caster == NDArray:
        #     type_caster = gmx.datamodel.ndarray
        # self._data[name] = Output(name=name,
        #                           dtype=type_annotation,
        #                           done=True,
        #                           data=type_caster(value))
        self._data[name].set(value)

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
                        # self._runner(*input.args, output=output, **input.kwargs)
                        ####
                        # Here we can make _runner a thing that accepts session resources, and
                        # is created by specializable builders. Separate out the expression of
                        # inputs.
                        #
                        # resource_builder = ResourcesBuilder()
                        # runner_builder = RunnerBuilder()
                        # input_resource_director = self._input_resource_factory.director(input)
                        # output_resource_director = self._publishing_resource_factory.director(output)
                        # input_resource_director(resource_builder, runner_builder)
                        # output_resource_director(resource_builder, runner_builder)
                        # resources = resource_builder.build()
                        # runner = runner_builder.build()
                        # runner(resources)
                        resources = self._operation.resource_director(input=input, output=output)
                        self._runner(resources)

    def future(self, name: str = None, description: gmx.datamodel.ResultDescription = None):
        """Retrieve a Future for a named output.

        Provide a description of the expected result to check for compatibility or
        implicit topological conversion.

        TODO: (FR5+) Normalize this part of the interface between operation definitions and
         resource managers.
        """
        if not isinstance(name, str) or name not in self._data:
            raise exceptions.ValueError('"name" argument must name an output.')
        assert description is not None
        requested_dtype = description.dtype
        available_dtype = self._data[name]._description.dtype
        if requested_dtype != available_dtype:
            # TODO: framework to check for implicit conversions
            message = 'Requested Future of type {} is not compatible with available type {}.'
            message = message.format(requested_dtype, available_dtype)
            raise exceptions.ApiError(message)
        return Future(self, name, description)

    def data(self):
        """Get an adapter to the output resources to access results."""
        return self._operation.OutputDataProxy(self)

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
        #  This should probably also use a sort of Context-based observer pattern rather than
        #  the result() method, which is explicitly for moving data across the API boundary.
        args = []
        try:
            for arg in self._input_fingerprint.args:
                value = arg
                if hasattr(value, 'result'):
                    value = value.result()
                if isinstance(value, NDArray):
                    value = value.values
                args.append(value)
        except Exception as E:
            raise exceptions.ApiError('input_fingerprint not iterating on "args" attr as expected.') from E

        kwargs = {}
        try:
            for key, value in self._input_fingerprint.kwargs.items():
                if hasattr(value, 'result'):
                    kwargs[key] = value.result()
                else:
                    kwargs[key] = value
                if isinstance(kwargs[key], NDArray):
                    kwargs[key] = kwargs[key].values
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
        return self.__publishing_resources.pop()()

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


# TODO: For outputs, distinguish between "results" and "events".
#  Both are published to the resource manager in the same way, but the relationship
#  with subscribers is potentially different.
def function_wrapper(output: dict = None):
    """Generate a decorator for wrapped functions with signature manipulation.

    New function accepts the same arguments, with additional arguments required by
    the API.

    The new function returns an object with an `output` attribute containing the named outputs.

    Example:
        @function_wrapper(output={'spam': str, 'foo': str})
        def myfunc(parameter: str = None, output=None):
            output.spam = parameter
            output.foo = parameter + ' ' + parameter

        operation1 = myfunc(parameter='spam spam')
        assert operation1.output.spam.result() == 'spam spam'
        assert operation1.output.foo.result() == 'spam spam spam spam'

    If 'output' is provided to the wrapper, a data structure will be passed to
    the wrapped functions with the named attributes so that the function can easily
    publish multiple named results. Otherwise, the `output` of the generated operation
    will just capture the return value of the wrapped function.
    """

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

    def decorator(function):

        class CapturedOutputRunner(object):
            """Function runner that captures return value as output.data"""

            def __init__(self, function):
                assert callable(function)
                self.function = function
                self.capture_output = None

            def __call__(self, resources):
                self.capture_output(self.function(*resources.args, **resources.kwargs))

        class OutputParameterRunner(object):
            """Function runner that uses output parameter to let function publish output."""

            def __init__(self, function):
                assert callable(function)
                self.function = function

            def __call__(self, resources):
                self.function(*resources.args, **resources.kwargs)

        class OperationDetails(object):
            """Manage the implementation details of an operation instance.

            Implementation is a Python function with resources managed by a
            resource manager.

            An OperationDetails instance should be owned by the resource manager
            rather than being directly owned by the client through an Operation
            handle.
            """

            def __init__(self, function=None, output_description=None):
                assert callable(function)
                signature = inspect.signature(function)

                # Determine output details
                # TODO FR4: standardize typing
                if 'output' in signature.parameters:
                    self.runner = OutputParameterRunner(function)
                    if not isinstance(output_description, gmx.datamodel.OutputCollectionDescription):
                        output_description = gmx.datamodel.OutputCollectionDescription(**output_description)
                else:
                    self.runner = CapturedOutputRunner(function)
                    # Use return type inferred from function signature as a hint.
                    return_type = signature.return_annotation
                    if isinstance(output_description, gmx.datamodel.OutputCollectionDescription):
                        return_type = self.output_description['data'].gmxapi_datatype
                    elif output_description is not None:
                        # output_description should be None for infered output or
                        # a singular mapping of the key 'data' to a gmxapi type.
                        if not isinstance(output_description, collections.abc.Mapping) \
                                or set(output_description.keys()) != {'data'}:
                            raise exceptions.ApiError(
                                'invalid output description for wrapped function: {}'.format(output_description))
                        if return_type == signature.empty:
                            return_type = output_description['data']
                        else:
                            if return_type != output_description['data']:
                                raise exceptions.ApiError(
                                    'Wrapped function with return-value-capture provided with non-matching output description.')
                    if return_type == signature.empty or return_type is None:
                        raise exceptions.ApiError('No return annotation for {}'.format(function))
                    output_description = gmx.datamodel.OutputCollectionDescription(data=return_type)
                self.output_description = output_description
                self._output_data_proxy = define_output_data_proxy(self.output_description)
                self._publishing_data_proxy = define_publishing_data_proxy(self.output_description)

                # Deterimine input details
                # TODO FR4: standardize typing

            def make_datastore(self):
                datastore = {}
                for name, dtype in self.output_description.items():
                    assert isinstance(dtype, type)
                    result_description = gmx.datamodel.ResultDescription(dtype)
                    datastore[name] = OutputData(name=name, description=result_description)
                return datastore

            @property
            def OutputDataProxy(self):
                return self._output_data_proxy

            @property
            def PublishingDataProxy(self):
                return self._publishing_data_proxy

            def resource_director(self, input=None, output=None):
                """a Director factory that helps build the Session Resources for the function."""
                resources = collections.namedtuple('Resources', ('args', 'kwargs'))([], {})
                resources.args.extend(input.args)
                resources.kwargs.update(input.kwargs)
                if not hasattr(self.runner, 'capture_output'):
                    resources.kwargs.update({'output': output})
                else:
                    # Bind the runner's return value capture to the `data` member of `output`
                    def capture(data):
                        output.data = data

                    self.runner.capture_output = capture
                return resources

        def get_resource_manager(instance):
            """Provide a reference to a resource manager for the dynamically defined Operation.

            Initial Operation implementation must own ResourceManager. As more formal Context is
            developed, this can be changed to a weak reference. A distinction can also be developed
            between the facet of the Context-level resource manager to which the Operation has access
            and the whole of the managed resources.
            """
            return ResourceManager(input_fingerprint=instance._input,
                                   operation=OperationDetails(function, output))

        @functools.wraps(function)
        def factory(**kwargs):
            signature = inspect.signature(function)

            class Operation(object):
                """Dynamically defined Operation implementation.

                Define a gmxapi Operation for the functionality being wrapped by the enclosing code.
                """

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
                    #
                    # Define the unique identity and data flow constraints of this work graph node.
                    #
                    # TODO: (FR4) generalize
                    input_dependencies = []

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
                    assert 'output' not in input_kwargs

                    # Merge kwargs and kwargs['input'] (keyword parameters versus parameter pack)
                    for key in kwargs:
                        if key in signature.parameters:
                            input_kwargs[key] = kwargs[key]
                        else:
                            raise exceptions.UsageError('Unexpected keyword argument: {}'.format(key))

                    # TODO: (FR4) Check input types

                    self.__input = PyFuncInput(args=[],
                                               kwargs=input_kwargs,
                                               dependencies=input_dependencies)

                    # TODO: (FR5+) Split the definition of the resource structure
                    #  and the resource initialization.
                    # Resource structure definition logic can be moved to the level
                    # of the class definition. We need knowledge of the inputs to
                    # uniquely identify the resources for this operation instance.
                    # Implementation suggestion: Context-provided metaclass defines
                    # resource manager interface for this Operation. Factory function
                    # initializes compartmentalized resource management at object creation.
                    self.__resource_manager = get_resource_manager(self)

                @property
                def _input(self):
                    """Internal interface to support data flow and execution management."""
                    return self.__input

                @property
                def output(self):
                    # Note: if we define Operation classes exclusively in the scope
                    # of Context instances, we could elegantly have a single _resource_manager
                    # handle instance per Operation type per Context instance.
                    # That could make it easier to implement library-level optimizations
                    # for managing hardware resources or data placement for operations
                    # implemented in the same librarary. That would be well in the future,
                    # though, and could also be accomplished with other means,
                    # so here I'm assuming one resource manager handle instance
                    # per Operation handle instance.
                    #
                    # TODO: Allow both structured and singular output.
                    #  Either return self._resource_manager.data or self._resource_manager.data.output
                    # TODO: We can configure `output` as a data descriptor
                    #  instead of a property so that we can get more information
                    #  from the class attribute before creating an instance.
                    # The C++ equivalence would probably be a templated free function for examining traits.
                    return self.__resource_manager.data()

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

#
# @function_wrapper(output={'simulation_input': str})
# def read_tpr(tprfile: str = ''):
#     """Prepare simulation input pack from a TPR file."""
#     output.simulation_input = ''
