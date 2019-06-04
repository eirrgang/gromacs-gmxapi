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
           'function_wrapper',
           ]

import abc
import functools
import importlib.util
import inspect
import typing
import weakref
from contextlib import contextmanager

import gmxapi as gmx
from gmxapi import logger as root_logger
from gmxapi.datamodel import *

# Initialize module-level logger
logger = root_logger.getChild(__name__)
logger.info('Importing gmxapi.operation')


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

        for name, param in sig.parameters.items():
            assert not param.kind == param.POSITIONAL_ONLY
        bound_arguments = sig.bind(*args, **kwargs)
        wrapped_function = function_wrapper()(function)
        handle = wrapped_function(**bound_arguments.arguments)
        output = handle.output
        return output.data

    return new_function


class OutputCollectionDescription(collections.OrderedDict):
    def __init__(self, **kwargs):
        """Create the output description for an operation node from a dictionary of names and types."""
        outputs = []
        for name, flavor in kwargs.items():
            if not isinstance(name, str):
                raise exceptions.TypeError('Output descriptions are keyed by Python strings.')
            # Multidimensional outputs are explicitly NDArray
            if issubclass(flavor, (list, tuple)):
                flavor = NDArray
            assert issubclass(flavor, (str, bool, int, float, dict, NDArray))
            outputs.append((name, flavor))
        super().__init__(outputs)


class InputCollectionDescription(collections.OrderedDict):
    """Describe acceptable inputs for an Operation.

    Generally, an InputCollectionDescription is an aspect of the public API by
    which an Operation expresses its possible inputs. This class includes details
    of the Python package.

    Keyword Arguments:
        parameters : A sequence of named parameter descriptions.

    Parameter descriptions are objects containing an `annotation` attribute
    declaring the data type of the parameter and, optionally, a `default`
    attribute declaring a default value for the parameter.

    Instances can be used as an ordered map of parameter names to gmxapi data types.

    Analogous to inspect.Signature, but generalized for gmxapi Operations.
    Additional notable differences: typing is normalized at initialization, and
    the bind() method does not return an object that can be directly used as
    function input. The object returned from bind() is used to construct a data
    graph Edge for subsequent execution.
    """

    def __init__(self, **parameters):
        """Create the input description for an operation node from a dictionary of names and types."""
        inputs = []
        for name, param in parameters.items():
            if not isinstance(name, str):
                raise exceptions.TypeError('Input descriptions are keyed by Python strings.')
            # Multidimensional inputs are explicitly NDArray
            dtype = param.annotation
            if issubclass(dtype, collections.abc.Iterable) \
                    and not issubclass(dtype, (str, bytes, collections.abc.Mapping)):
                # TODO: we can relax this with some more input conditioning.
                if dtype != NDArray:
                    raise exceptions.UsageError(
                        'Cannot accept input type {}. Sequence type inputs must use NDArray.'.format(param))
            assert issubclass(dtype, (str, bool, int, float, dict, NDArray))
            if hasattr(param, 'kind'):
                disallowed = any([param.kind == param.POSITIONAL_ONLY,
                                  param.kind == param.VAR_POSITIONAL,
                                  param.kind == param.VAR_KEYWORD])
                if disallowed:
                    raise exceptions.ProtocolError(
                        'Cannot wrap function. Operations must have well-defined parameter names.')
                kind = param.kind
            else:
                kind = inspect.Parameter.KEYWORD_ONLY
            if hasattr(param, 'default'):
                default = param.default
            else:
                default = inspect.Parameter.empty
            inputs.append(inspect.Parameter(name, kind, default=default, annotation=dtype))
        super().__init__([(input.name, input.annotation) for input in inputs])
        self.signature = inspect.Signature(inputs)

    @staticmethod
    def from_function(function):
        """Inspect a function to be wrapped.

        Used internally by gmxapi.operation.function_wrapper()

            Raises:
                exceptions.ProtocolError if function signature cannot be determined to be valid.

            Returns:
                InputCollectionDescription for the function input signature.
        """
        # First, inspect the function.
        assert callable(function)
        signature = inspect.signature(function)
        # The function must have clear and static input schema
        # Make sure that all parameters have clear names, whether or not they are used in a call.
        for name, param in signature.parameters.items():
            disallowed = any([param.kind == param.POSITIONAL_ONLY,
                              param.kind == param.VAR_POSITIONAL,
                              param.kind == param.VAR_KEYWORD])
            if disallowed:
                raise exceptions.ProtocolError(
                    'Cannot wrap function. Operations must have well-defined parameter names.')
            if param.name == 'input':
                raise exceptions.ProtocolError('Function signature includes the (reserved) "input" keyword argument.')
        description = collections.OrderedDict()
        for param in signature.parameters.values():
            if param.name == 'output':
                # Wrapped functions may accept the output parameter to publish results, but
                # that is not part of the Operation input signature.
                continue
            if param.annotation == param.empty:
                if param.default == param.empty or param.default is None:
                    raise exceptions.ProtocolError('Could not infer parameter type for {}'.format(param.name))
                dtype = type(param.default)
                if isinstance(dtype, collections.abc.Iterable) \
                        and not isinstance(dtype, (str, bytes, collections.abc.Mapping)):
                    dtype = NDArray
            else:
                dtype = param.annotation
            description[param.name] = param.replace(annotation=dtype)
        return InputCollectionDescription(**description)

    def bind(self, *args, **kwargs) -> DataSourceCollection:
        """Create a compatible DataSourceCollection from provided arguments.

        Pre-process input and function signature to get named input arguments.

        This is a helper function to allow calling code to characterize the
        arguments in a Python function call with hints from the factory that is
        initializing an operation. Its most useful functionality is to  allows a
        factory to accept positional arguments where named inputs are usually
        required. It also allows data sources to participate in multiple
        DataSourceCollections with minimal constraints.

        Note that the returned object has had data populated from any defaults
        described in the InputCollectionDescription.

        See wrapped_function_runner() and describe_function_input().
        """
        # For convenience, accept *args, but convert to **kwargs to pass to Operation.
        # Factory accepts an unadvertised `input` keyword argument that is used as a default kwargs dict.
        # If present, kwargs['input'] is treated as an input "pack" providing _default_ values.
        input_kwargs = collections.OrderedDict()
        # Note: we have also been allowing arguments with a `run` attribute
        # to be used as execution dependencies, but we should probably stop that.
        # TODO: (FR4) generalize
        execution_dependencies = []
        if 'input' in kwargs:
            provided_input = kwargs.pop('input')
            if provided_input is not None:
                # Note: we have also been allowing arguments with a `run` attribute
                # to be used as execution dependencies, but we should probably stop that.
                if hasattr(provided_input, 'run'):
                    execution_dependencies.append(provided_input)
                    # Note that execution_dependencies is not used after this point.
                else:
                    input_kwargs.update(provided_input)
        # `function` may accept an `output` keyword argument that should not be supplied to the factory.
        for key, value in kwargs.items():
            if key == 'output':
                raise exceptions.UsageError('Invalid keyword argument: output (reserved).')
            input_kwargs[key] = value
        try:
            bound_arguments = self.signature.bind_partial(*args, **input_kwargs)
        except TypeError as e:
            raise exceptions.UsageError('Could not bind operation parameters to function signature.') from e
        assert 'output' not in bound_arguments.arguments
        bound_arguments.apply_defaults()
        assert 'input' not in bound_arguments.arguments
        input_kwargs = collections.OrderedDict([pair for pair in bound_arguments.arguments.items()])
        if 'output' in input_kwargs:
            input_kwargs.pop('output')
        return DataSourceCollection(**input_kwargs)


class ProxyDataDescriptor(object):
    """Base class for data descriptors used in DataProxyBase subclasses.

    Subclasses should either not define __init__ or should call the base class
    __init__ explicitly: super().__init__(self, name, dtype)
    """

    def __init__(self, name: str, dtype=None):
        self._name = name
        if dtype is not None:
            assert isinstance(dtype, type)
            assert issubclass(dtype, (str, bool, int, float, dict, NDArray))
        self._dtype = dtype


class DataProxyMeta(type):
    # Key word arguments consumed by __prepare__
    _prepare_keywords = ('descriptors',)

    @classmethod
    def __prepare__(mcs, name, bases, descriptors: collections.abc.Mapping = None):
        """Allow dynamic sub-classing.

        DataProxy class definitions are collections of data descriptors. This
        class method allows subclasses to give the descriptor names and type(s)
        in the class declaration as arguments instead of as class attribute
        assignments.

            class MyProxy(DataProxyBase, descriptors={name: MyDescriptor() for name in datanames}): pass

        Note:
            If we are only using this metaclass for the __prepare__ hook by the
            time we require Python >= 3.6, we could reimplement __prepare__ as
            DataProxyBase.__init_subclass__ and remove this metaclass.
        """
        if descriptors is None:
            return {}
        else:
            return descriptors

    def __new__(cls, name, bases, namespace, **kwargs):
        for key in kwargs:
            if key not in DataProxyMeta._prepare_keywords:
                raise exceptions.ApiError('Unexpected class creation keyword: {}'.format(key))
        return type.__new__(cls, name, bases, namespace)

    # TODO: This is keyword argument stripping is not necessary in more recent Python versions.
    # When Python minimum required version is increased, check if we can remove this.
    def __init__(cls, name, bases, namespace, **kwargs):
        for key in kwargs:
            if key not in DataProxyMeta._prepare_keywords:
                raise exceptions.ApiError('Unexpected class initialization keyword: {}'.format(key))
        super().__init__(name, bases, namespace)


class DataProxyBase(object, metaclass=DataProxyMeta):
    """Limited interface to managed resources.

    Inherit from DataProxy to specialize an interface to an ``instance``.
    In the derived class, either do not define ``__init__`` or be sure to
    initialize the super class (DataProxy) with an instance of the object
    to be proxied.

    Acts as an owning handle to ``instance``, preventing the reference count
    of ``instance`` from going to zero for the lifetime of the proxy object.

    When sub-classing DataProxyBase, data descriptors can be passed as a mapping
    to the ``descriptors`` key word argument in the class declaration. This
    allows data proxy subclasses to be easily defined dynamically.

        mydescriptors = {'foo': Publisher('foo', int), 'data': Publisher('data', float)}
        ...
        class MyDataProxy(DataProxyBase, descriptors=mydescriptors): pass
        assert hasattr(MyDataProxy, 'foo')

    """

    # This class can be expanded to be the attachment point for a metaclass for
    # data proxies such as PublishingDataProxy or OutputDataProxy, which may be
    # defined very dynamically and concisely as a set of Descriptors and a type()
    # call.
    # If development in this direction does not materialize, then this base
    # class is not very useful and should be removed.
    def __init__(self, instance, client_id: int = None):
        """Get partial ownership of a resource provider.

        Arguments:
            instance : resource-owning object
            client_id : identifier for client holding the resource handle (e.g. ensemble member id)

        If client_id is not provided, the proxy scope is for all clients.
        """
        # Developer note subclasses should handle self._client_identifier == None
        self._resource_instance = instance
        self._client_identifier = client_id

    @property
    def ensemble_width(self):
        return self._resource_instance.ensemble_width

    @classmethod
    def items(cls):
        """Generator for tuples of attribute name and descriptor instance.

        This almost certainly doesn't do quite what we want...
        """
        for name, value in cls.__dict__.items():
            if isinstance(value, ProxyDataDescriptor):
                yield name, value


class Publisher(ProxyDataDescriptor):
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

    def __get__(self, instance: DataProxyBase, owner):
        if instance is None:
            # Access through class attribute of owner class
            return self
        resource_manager = instance._resource_instance
        client_id = instance._client_identifier
        if client_id is None:
            return getattr(resource_manager._data, self._name)
        else:
            return getattr(resource_manager._data, self._name)[client_id]

    def __set__(self, instance: DataProxyBase, value):
        resource_manager = instance._resource_instance
        client_id = instance._client_identifier
        resource_manager.set_result(name=self._name, value=value, member=client_id)

    def __repr__(self):
        return 'Publisher(name={}, dtype={})'.format(self._name, self._dtype.__qualname__)


def define_publishing_data_proxy(output_description) -> typing.Type[DataProxyBase]:
    """Returns a class definition for a PublishingDataProxy for the provided output description."""
    # This dynamic type creation hides collaborations with things like make_datastore.
    # We should encapsulate these relationships in Context details, explicit collaborations
    # between specific operations and Contexts, and in groups of Operation definition helpers.

    descriptors = collections.OrderedDict([(name, Publisher(name)) for name in output_description])

    class PublishingDataProxy(DataProxyBase, descriptors=descriptors):
        """Handler for write access to the `output` of an operation.

        Acts as a sort of PublisherCollection.
        """

    return PublishingDataProxy


class SourceResource(abc.ABC):
    """Resource Manager for a data provider."""

    # @classmethod
    # def __subclasshook__(cls, C):
    #     """Check if a class looks like a ResourceManager for a data source."""
    #     if cls is SourceResource:
    #         if any('update_output' in B.__dict__ for B in C.__mro__) \
    #                 and hasattr(C, '_data'):
    #             return True
    #     return NotImplemented

    # This might not belong here. Maybe separate out for a OperationHandleManager?
    @abc.abstractmethod
    def data(self) -> DataProxyBase:
        """Get the output data proxy."""
        ...

    @abc.abstractmethod
    def is_done(self, name: str) -> bool:
        return False

    @abc.abstractmethod
    def get(self, name: str) -> 'OutputData':
        ...

    @abc.abstractmethod
    def update_output(self):
        """Bring the _data member up to date and local."""
        pass

    @abc.abstractmethod
    def reset(self):
        """Recursively reinitialize resources.

        Set the resource manager to its initialized state.
        All outputs are marked not "done".
        All inputs supporting the interface have ``_reset()`` called on them.
        """


class StaticSourceManager(SourceResource):
    """Provide the resource manager interface for local static data.

    Allow data transformations on the proxied resource.

    Keyword Args:
        proxied_data: A gmxapi supported data object.
        width: Size of (one-dimensional) shaped data produced by function.
        function: Transformation to perform on the managed data.

    The callable passed as ``function`` must accept a single argument. The
    argument will be an iterable when proxied_data represents an ensemble,
    or an object of the same type as proxied_data otherwise.
    """

    def __init__(self, *, name: str, proxied_data, width: int, function: typing.Callable):
        assert not isinstance(proxied_data, Future)
        if hasattr(proxied_data, 'width'):
            # Ensemble data source
            assert hasattr(proxied_data, 'source')
            self._result = function(proxied_data.source)
        else:
            self._result = function(proxied_data)
        if width > 1:
            if isinstance(self._result, (str, bytes)):
                # In this case, do not implicitly broadcast
                raise exceptions.ValueError('"function" produced data incompatible with "width".')
            else:
                if not isinstance(self._result, collections.abc.Iterable):
                    raise exceptions.DataShapeError(
                        'Expected iterable of size {} but "function" result is not iterable.')
            data = list(self._result)
            size = len(data)
            if len(data) != width:
                raise exceptions.DataShapeError(
                    'Expected iterable of size {} but "function" produced a {} of size {}'.format(width, type(data),
                                                                                                  size))
            dtype = type(data[0])
        else:
            if width != 1:
                raise exceptions.ValueError('width must be an integer 1 or greater.')
            dtype = type(self._result)
            if issubclass(dtype, (list, tuple)):
                dtype = NDArray
                data = [ndarray(self._result)]
            elif isinstance(self._result, collections.abc.Iterable):
                if not isinstance(self._result, (str, bytes)):
                    raise exceptions.ValueError(
                        'Expecting width 1 but "function" produced iterable type {}.'.format(type(self._result)))
                else:
                    dtype = str
                    data = [str(self._result)]
            else:
                data = [self._result]
        description = ResultDescription(dtype=dtype, width=width)
        self._data = OutputData(name=name, description=description)
        for member in range(width):
            self._data.set(data[member], member=member)

        output_collection_description = OutputCollectionDescription(**{name: dtype})
        self.output_data_proxy = define_output_data_proxy(output_description=output_collection_description)

    def is_done(self, name: str) -> bool:
        return True

    def get(self, name: str) -> 'OutputData':
        assert self._data.name == name
        return self._data

    def data(self) -> DataProxyBase:
        return self.output_data_proxy(self)

    def update_output(self):
        pass

    def reset(self):
        pass


class ProxyResourceManager(SourceResource):
    """Act as a resource manager for data managed by another resource manager.

    Allow data transformations on the proxied resource.

    Keyword Args:
        proxied_future: An object implementing the Future interface.
        width: Size of (one-dimensional) shaped data produced by function.
        function: Transformation to perform on the result of proxied_future.

    The callable passed as ``function`` must accept a single argument, which will
    be an iterable when proxied_future represents an ensemble, or an object of
    type proxied_future.description.dtype otherwise.
    """

    def __init__(self, proxied_future: 'Future', width: int, function: typing.Callable):
        self._done = False
        self._proxied_future = proxied_future
        self._width = width
        self.name = self._proxied_future.name
        self._result = None
        assert callable(function)
        self.function = function

    def reset(self):
        self._done = False
        self._proxied_future._reset()
        self._result = None

    def is_done(self, name: str) -> bool:
        return self._done

    def get(self, name: str):
        if name != self.name:
            raise exceptions.ValueError('Request for unknown data.')
        if not self.is_done(name):
            raise exceptions.ProtocolError('Data not ready.')
        result = self.function(self._result)
        if self._width != 1:
            # TODO Fix this typing nightmare:
            #  ResultDescription should be fully knowable and defined when the resource manager is initialized.
            data = OutputData(name=self.name, description=ResultDescription(dtype=type(result[0]), width=self._width))
            for member, value in enumerate(result):
                data.set(value, member)
        else:
            data = OutputData(name=self.name, description=ResultDescription(dtype=type(result), width=self._width))
            data.set(result, 0)
        return data

    def update_output(self):
        self._result = self._proxied_future.result()
        self._done = True

    def data(self) -> DataProxyBase:
        raise exceptions.ApiError('ProxyResourceManager cannot yet manage a full OutputDataProxy.')


class AbstractOperationHandle(abc.ABC):
    """Handle to an operation instance (graph node)."""

    @abc.abstractmethod
    def run(self):
        """Assert execution of an operation.

        After calling run(), the operation results are guaranteed to be available
        in the local context.
        """

    @property
    @abc.abstractmethod
    def output(self) -> DataProxyBase:
        """Get a proxy collection to the output of the operation."""
        ...

    # TODO: Factory for translating an operation from one context to another.
    # E.g.
    #
    #     @classmethod
    #     @abc.abstractmethod
    #     def factory(cls, context=None, input: typing.Mapping = None) -> 'AbstractOperationHandle':
    #         """Dispatch an Operation factory for the given Context and input."""
    #         ...


# TODO: Implement observer pattern for edge->node data flow.
# Step 0: implement subject interface subscribe()
# Step 1: implement subject interface get_state()
# Step 2: implement observer interface update()
# Step 3: implement subject interface notify()
# class Observer(object):
#     """Abstract base class for data observers."""
#     def rebind(self, edge: DataEdge):
#         """Recreate the Operation at the consuming end of the DataEdge."""


class Future(object):
    """gmxapi data handle.

    Future is currently more than a Future right now. (should be corrected / clarified.)
    Future is also a facade to other features of the data provider. The ``subscribe``
    method allows consumers to bind as Observers.

    TODO: extract the abstract class for input inspection?
    Currently abstraction is handled through SourceResource subclassing.
    """

    def __init__(self, resource_manager: SourceResource, name: str, description: gmx.datamodel.ResultDescription):
        self.name = name
        if not isinstance(description, gmx.datamodel.ResultDescription):
            raise exceptions.ValueError('Need description of requested data.')
        self.resource_manager = resource_manager
        self.description = description

    def result(self):
        """Fetch data to the caller's Context.

        Returns an object of the concrete type specified according to
        the operation that produces this Result.
        """
        self.resource_manager.update_output()
        assert self.resource_manager.is_done(self.name)
        # Return ownership of concrete data
        handle = self.resource_manager.get(self.name)
        return handle.data

    def _reset(self):
        """Mark the Future "not done" to allow reexecution.

        Invalidates cached results, resets "done" markers in data sources, and
        triggers _reset recursively.

        Note: this is a hack that is inconsistent with the plan of unique mappings
        of inputs to outputs, but allows a quick prototype for looping operations.
        """
        self.resource_manager.reset()

    @property
    def dtype(self):
        return self.description.dtype

    def __getitem__(self, item):
        """Get a more limited view on the Future."""
        description = gmx.datamodel.ResultDescription(dtype=self.dtype, width=self.description.width)
        # TODO: Use explicit typing when we have more thorough typing.
        description._dtype = None
        if self.description.width == 1:
            proxy = ProxyResourceManager(self,
                                         width=description.width,
                                         function=lambda value, key=item: value[key])
        else:
            proxy = ProxyResourceManager(self,
                                         width=description.width,
                                         function=lambda value, key=item:
                                         [subscriptable[key] for subscriptable in value])
        future = Future(proxy, self.name, description=description)
        return future


class OutputDescriptor(ProxyDataDescriptor):
    """Read-only data descriptor for proxied output access.

    Knows how to get a Future from the resource manager.
    """

    def __get__(self, proxy: DataProxyBase, owner):
        if proxy is None:
            # Access through class attribute of owner class
            return self
        result_description = gmx.datamodel.ResultDescription(dtype=self._dtype, width=proxy.ensemble_width)
        return proxy._resource_instance.future(name=self._name, description=result_description)


def define_output_data_proxy(output_description: OutputCollectionDescription) -> typing.Type[DataProxyBase]:
    descriptors = {name: OutputDescriptor(name, description) for name, description in output_description.items()}

    class OutputDataProxy(DataProxyBase, descriptors=descriptors):
        """Handler for read access to the `output` member of an operation handle.

        Acts as a sort of ResultCollection.

        A ResourceManager creates an OutputDataProxy instance at initialization to
        provide the ``output`` property of an operation handle.
        """

    # Note: the OutputDataProxy has an inherent ensemble shape in the context
    # in which it is used, but that is an instance characteristic, not part of this type definition.
    # TODO: (FR5) The current tool does not support topology changing operations.
    return OutputDataProxy


# In the longer term, Contexts could provide metaclasses that allow transformation or dispatching
# of the basic aspects of the operation protocols between Contexts or from a result handle into a
# new context, based on some attribute or behavior in the result handle.


# Encapsulate the description of the input data flow.
PyFuncInput = collections.namedtuple('Input', ('args', 'kwargs', 'dependencies'))


# Encapsulate the description and storage of a data output.
class OutputData(object):
    def __init__(self, name: str, description: gmx.datamodel.ResultDescription):
        assert name != ''
        self._name = name
        assert isinstance(description, gmx.datamodel.ResultDescription)
        self._description = description
        self._done = [False] * self._description.width
        self._data = [None] * self._description.width

    @property
    def name(self):
        return self._name

    # TODO: Change to regular member function and add ensemble member arg.
    @property
    def done(self):
        return all(self._done)

    # TODO: Change to regular member function and add ensemble member arg.
    @property
    def data(self):
        if not self.done:
            raise exceptions.ApiError('Attempt to read before data has been published.')
        if self._data is None or None in self._data:
            raise exceptions.ApiError('Data marked "done" but contains null value.')
        # For intuitive use in non-ensemble cases, we represent data as bare scalars
        # when possible. It is easy to cast scalars to lists of length 1. In the future,
        # we may distinguish between data of shape () and shape (1,), but we will need
        # to be careful with semantics. We are already starting to adopt a rule-of-thumb
        # that data objects assume the minimum dimensionality necessary unless told
        # otherwise, and we could make that a hard rule if it doesn't make other things
        # too difficult.
        if self._description.width == 1:
            return self._data[0]
        else:
            return self._data

    def set(self, value, member: int):
        if self._description.dtype == NDArray:
            self._data[member] = gmx.datamodel.ndarray(value)
        else:
            self._data[member] = self._description.dtype(value)
        self._done[member] = True

    def reset(self):
        self._done = [False] * self._description.width
        self._data = [None] * self._description.width


class SinkTerminal(object):
    """Operation input end of a data edge.

    In addition to the information in an InputCollectionDescription, includes
    topological information for the Operation node (ensemble width).

    Collaborations: Required for creation of a DataEdge. Created with knowledge
    of a DataSourceCollection instance and a InputCollectionDescription.
    """

    # TODO: This clearly maps to a Builder pattern.
    # I think we want to get the sink terminal builder from a factory parameterized by InputCollectionDescription,
    # add data source collections, and then build the sink terminal for the data edge.
    def __init__(self, input_collection_description: InputCollectionDescription):
        """Define an appropriate data sink for a new operation node.

        Resolve data sources and input description to determine connectability,
        topology, and any necessary implicit data transformations.

        :param data_source_collection: Collection of offered input data.
        :param input_collection_description: Available inputs for Operation
        :return: Fully formed description of the Sink terminal for a data edge to be created.

        Collaborations: Execution Context implementation.
        """
        self.ensemble_width = 1
        self.inputs = input_collection_description

    def update_width(self, width: int):
        if not isinstance(width, int):
            try:
                width = int(width)
            except TypeError:
                raise exceptions.TypeError('Need an integer width > 0.')
        if width < 1:
            raise exceptions.ValueError('Nonsensical ensemble width: {}'.format(int(width)))
        if self.ensemble_width != 1:
            if width != self.ensemble_width:
                raise exceptions.ValueError(
                    'Cannot change ensemble width {} to width {}.'.format(self.ensemble_width, width))
        self.ensemble_width = width

    def update(self, data_source_collection: DataSourceCollection):
        """Update the SinkTerminal with the proposed data provider."""
        for name, sink_dtype in self.inputs.items():
            if name not in data_source_collection:
                # If/when we accept data from multiple sources, we'll need some additional sanity checking.
                if not hasattr(self.inputs.signature.parameters[name], 'default'):
                    raise exceptions.UsageError('No data or default for {}'.format(name))
            else:
                # With a single data source, we need data to be in the source or have a default
                assert name in data_source_collection
                assert issubclass(sink_dtype, (str, bool, int, float, dict, NDArray))
                source = data_source_collection[name]
                if isinstance(source, sink_dtype):
                    continue
                else:
                    if isinstance(source, collections.abc.Iterable) and not isinstance(source, (
                            str, bytes, collections.abc.Mapping)):
                        assert isinstance(source, NDArray)
                        if sink_dtype != NDArray:
                            # Implicit scatter
                            self.update_width(len(source))


class DataEdge(object):
    """State and description of a data flow edge.

    A DataEdge connects a data source collection to a data sink. A sink is an
    input or collection of inputs of an operation (or fused operation). An operation's
    inputs may be fed from multiple data source collections, but an operation
    cannot be fully instantiated until all of its inputs are bound, so the DataEdge
    is instantiated at the same time the operation is instantiated because the
    required topology of a graph edge may be determined by the required topology
    of another graph edge.

    A data edge has a well-defined topology only when it is terminated by both
    a source and sink. Creation requires that a source collection is compared to
    a sink description.

    Calling code initiates edge creation by passing well-described data sources
    to an operation factory. The data sources may be annotated with explicit scatter
    or gather commands.

    The resource manager for the new operation determines the
    required shape of the sink to handle all of the offered input.

    Broadcasting
    and transformations of the data sources are then determined and the edge is
    established.

    At that point, the fingerprint of the input data at each operation
    becomes available to the resource manager for the operation. The fingerprint
    has sufficient information for the resource manager of the operation to
    request and receive data through the execution context.

    Instantiating operations and data edges implicitly involves collaboration with
    a Context instance. The state of a given Context or the availability of a
    default Context through a module function may affect the ability to instantiate
    an operation or edge. In other words, behavior may be different for connections
    being made in the scripting environment versus the running Session, and implementation
    details can determine whether or not new operations or data flow can occur in
    different code environments.
    """

    class ConstantResolver(object):
        def __init__(self, value):
            self.value = value

        def __call__(self, member=None):
            return self.value

    def __init__(self, source_collection: DataSourceCollection, sink_terminal: SinkTerminal):
        # Adapters are callables that transform a source and node ID to local data.
        # Every key in the sink has an adapter.
        self.adapters = {}
        self.source_collection = source_collection
        self.sink_terminal = sink_terminal
        for name in sink_terminal.inputs:
            if name not in source_collection:
                if hasattr(sink_terminal.inputs[name], 'default'):
                    self.adapters[name] = self.ConstantResolver(sink_terminal.inputs[name])
                else:
                    # TODO: Initialize with multiple DataSourceCollections?
                    raise exceptions.ValueError('No source or default for required input "{}".'.format(name))
            else:
                source = source_collection[name]
                sink = sink_terminal.inputs[name]
                if isinstance(source, (str, bool, int, float, dict)):
                    if issubclass(sink, (str, bool, int, float, dict)):
                        self.adapters[name] = self.ConstantResolver(source)
                    else:
                        assert issubclass(sink, NDArray)
                        self.adapters[name] = self.ConstantResolver(ndarray([source]))
                elif isinstance(source, NDArray):
                    if issubclass(sink, NDArray):
                        # TODO: shape checking
                        # Implicit broadcast may not be what is intended
                        self.adapters[name] = self.ConstantResolver(source)
                    else:
                        if source.shape[0] != sink_terminal.ensemble_width:
                            raise exceptions.ValueError(
                                'Implicit broadcast could not match array source to ensemble sink')
                        else:
                            self.adapters[name] = lambda member, source=source: source[member]
                elif hasattr(source, 'result'):
                    # Handle data futures...
                    # If the Future is part of an ensemble, result() will return a list.
                    # Otherwise, it will return a single object.
                    ensemble_width = source.description.width
                    if ensemble_width == 1:
                        self.adapters[name] = lambda member, source=source: source.result()
                    else:
                        self.adapters[name] = lambda member, source=source: source.result()[member]
                else:
                    assert isinstance(source, EnsembleDataSource)
                    self.adapters[name] = lambda member, source=source: source.node(member)

    def reset(self):
        self.source_collection.reset()

    def resolve(self, key: str, member: int):
        return self.adapters[key](member=member)

    def sink(self, node: int) -> dict:
        """Consume data for the specified sink terminal node.

        Run-time utility delivers data from the bound data source(s) for the
        specified terminal that was configured when the edge was created.

        Terminal node is identified by a member index number.

        Returns:
            A Python dictionary of the provided inputs as local data (not Future).
        """
        results = {}
        sink_ports = self.sink_terminal.inputs
        for key in sink_ports:
            results[key] = self.resolve(key, node)
        return results


class ResourceManager(SourceResource):
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
    def __publishing_context(self, ensemble_member=0):
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

        # I don't think we want the OperationDetails to need to know about ensemble data,
        # (though the should probably be allowed to), so we may need a separate interface
        # for the resource manager with built-in scope-limiting to a single ensemble member.
        # Right now, one Operation handle owns one ResourceManager (which takes care of
        # the ensemble details), which owns one OperationDetails (which has no ensemble knowledge).
        # It is the responsibility of the calling code to make sure the PublishingDataProxy
        # gets used correctly.

        # ref: https://docs.python.org/3/library/contextlib.html#contextlib.contextmanager
        try:
            if not self._done[ensemble_member]:
                resource = self._operation.publishing_data_proxy(weakref.proxy(self), ensemble_member)
                yield resource
        except Exception as e:
            message = 'Uncaught exception while providing output-publishing resources for {}.'.format(self._runner)
            raise exceptions.ApiError(message) from e
        finally:
            self._done[ensemble_member] = True

    def __init__(self, source: DataEdge = None, operation=None):
        """Initialize a resource manager for the inputs and outputs of an operation.

        Arguments:
            operation : implementation details for a Python callable
            input_fingerprint : Uniquely identifiable input data description

        """
        runner = operation.runner
        assert callable(runner)

        # Note: This implementation assumes there is one ResourceManager instance per data source,
        # so we only stash the inputs and dependency information for a single set of resources.
        # TODO: validate input_fingerprint as its interface becomes clear.
        self._input_edge = source
        self.ensemble_width = self._input_edge.sink_terminal.ensemble_width
        self._operation = operation

        self._data = self._operation.make_datastore(self.ensemble_width)

        # We store a rereference to the publishing context manager implementation
        # in a data structure that can only produce one per Python interpreter
        # (using list.pop()).
        # TODO: reimplement as a data descriptor
        #  so that PublishingDataProxy does not need a bound circular reference.
        self.__publishing_resources = [self.__publishing_context]

        self._done = [False] * self.ensemble_width
        self._runner = runner
        self.__operation_entrance_counter = 0

    def reset(self):
        self.__operation_entrance_counter = 0
        self._done = [False] * self.ensemble_width
        self.__publishing_resources = [self.__publishing_context]
        for data in self._data.values():
            data.reset()
        self._input_edge.reset()
        assert self.__operation_entrance_counter == 0

    def done(self, member=None):
        if member is None:
            return all(self._done)
        else:
            return self._done[member]

    def set_result(self, name, value, member: int):
        try:
            for item in value:
                # In this specification, it is antithetical to publish Futures.
                if hasattr(item, 'result'):
                    raise exceptions.ApiError('Operation produced Future instead of real output.')
        except TypeError:
            # Ignore when `item` is not iterable.
            pass
        self._data[name].set(value=value, member=member)

    def is_done(self, name):
        return self._data[name].done

    def get(self, name: str):
        """

        Raises exceptions.ProtocolError if requested data is not local yet.
        Raises exceptions.ValueError if data is requested for an unknown name.
        """
        if name not in self._data:
            raise exceptions.ValueError('Request for unknown data.')
        if not self.is_done(name):
            raise exceptions.ProtocolError('Data not ready.')
        assert isinstance(self._data[name], OutputData)
        return self._data[name]

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
        # TODO: Handle checking just the ensemble members this resource manager is responsible for.
        # TODO: Replace with a managed observer pattern. Update once when input is available in the Context.
        if not self.done():
            self.__operation_entrance_counter += 1
            if self.__operation_entrance_counter > 1:
                raise exceptions.ProtocolError('Bug detected: resource manager tried to execute operation twice.')
            if not self.done():
                # Note! This is a detail of the ResourceManager in a SerialContext
                for i in range(self.ensemble_width):
                    with self.local_input(i) as input:
                        # Note: Resources are marked "done" by the resource manager
                        # when the following context manager completes.
                        with self.publishing_resources()(ensemble_member=i) as output:
                            # self._runner(*input.args, output=output, **input.kwargs)
                            ####
                            # Here we can make _runner a thing that accepts session resources, and
                            # is created by specializable builders. Separate out the expression of
                            # inputs.
                            #
                            # resource_builder = OperationDetails.ResourcesBuilder(context)
                            # runner_builder = OperationDetails.RunnerBuilder(context)
                            # input_resource_director = self._input_resource_factory.director(input)
                            # output_resource_director = self._publishing_resource_factory.director(output)
                            # input_resource_director(resource_builder, runner_builder)
                            # output_resource_director(resource_builder, runner_builder)
                            # resources = resource_builder.build()
                            # runner = runner_builder.build()
                            # runner(resources)
                            resources = self._operation.resource_director(input=input, output=output)
                            self._runner(resources)

    def future(self, name: str, description: gmx.datamodel.ResultDescription):
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

    def data(self) -> DataProxyBase:
        """Get an adapter to the output resources to access results."""
        return self._operation.output_data_proxy(self)

    @contextmanager
    def local_input(self, member: int = None):
        """In an API session, get a handle to fully resolved locally available input data.

        Execution dependencies are resolved on creation of the context manager. Input data
        becomes available in the ``as`` object when entering the context manager, which
        becomes invalid after exiting the context manager. Resources allocated to hold the
        input data may be released when exiting the context manager.

        It is left as an implementation detail whether the context manager is reusable and
        under what circumstances one may be obtained.
        """
        # Localize data
        kwargs = self._input_edge.sink(node=member)
        assert 'input' not in kwargs

        # Check that we have real data
        for key, value in kwargs.items():
            assert not hasattr(value, 'result')
            assert not hasattr(value, 'run')
            value_list = []
            if isinstance(value, list):
                value_list = value
            if isinstance(value, NDArray):
                value_list = value._values
            if isinstance(value, collections.abc.Mapping):
                value_list = value.values()
            assert not isinstance(value_list, (Future))
            assert not hasattr(value_list, 'result')
            assert not hasattr(value_list, 'run')
            for item in value_list:
                assert not hasattr(item, 'result')

        input_pack = collections.namedtuple('InputPack', ('kwargs'))(kwargs)

        # Prepare input data structure
        # Note: we use 'yield' instead of 'return' for the protocol expected by
        # the @contextmanager decorator
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


class PyFunctionRunnerResources(collections.UserDict):
    """Runtime resources for Python functions.

    Produced by a ResourceDirector for a particular Operation.
    """

    def output(self):
        if 'output' in self:
            return self['output']
        else:
            return None

    def input(self):
        return {key: value for key, value in self.items() if key != 'output'}


class PyFunctionRunner(abc.ABC):
    def __init__(self, function, output_description: OutputCollectionDescription):
        assert callable(function)
        self.function = function
        self.output_description = output_description

    @abc.abstractmethod
    def __call__(self, resources: PyFunctionRunnerResources):
        self.function(output=resources.output(), **resources.input())


class CapturedOutputRunner(PyFunctionRunner):
    """Function runner that captures return value as output.data"""

    def __call__(self, resources: PyFunctionRunnerResources):
        resources['output'].data = self.function(**resources.input())


class OutputParameterRunner(PyFunctionRunner):
    """Function runner that uses output parameter to let function publish output."""

    def __call__(self, resources: PyFunctionRunnerResources):
        self.function(**resources)


def wrapped_function_runner(function, output_description: OutputCollectionDescription = None):
    """Get an adapter for a function to be wrapped.

    If the function does not accept a publishing data proxy as an `output`
    key word argument, the returned object has a `capture_output` attribute that
    must be re-assigned by the calling code before calling the runner. `capture_output`
    must be assigned to be a callable that will receive the output of the wrapped
    function.

    Returns:
        Callable with a signature `__call__(*args, **kwargs)` and no return value

    Collaborations:
        OperationDetails.resource_director assigns the `capture_output` member of the returned object.
    """
    assert callable(function)
    signature = inspect.signature(function)

    # Determine output details
    # TODO FR4: standardize typing
    if 'output' in signature.parameters:
        if not isinstance(output_description, OutputCollectionDescription):
            if not isinstance(output_description, collections.abc.Mapping):
                raise exceptions.UsageError(
                    'Function passes output through call argument, but output is not described.')
            return OutputParameterRunner(function, OutputCollectionDescription(**output_description))
        else:
            return OutputParameterRunner(function, output_description)
    else:
        # Use return type inferred from function signature as a hint.
        return_type = signature.return_annotation
        if isinstance(output_description, OutputCollectionDescription):
            return_type = output_description['data'].gmxapi_datatype
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
        return CapturedOutputRunner(function, OutputCollectionDescription(data=return_type))


class OperationHandle(AbstractOperationHandle):
    """Dynamically defined Operation handle.

    Define a gmxapi Operation for the functionality being wrapped by the enclosing code.

    An Operation type definition encapsulates description of allowed inputs
    of an Operation. An Operation instance represents a node in a work graph
    with uniquely fingerprinted inputs and well-defined output. The implementation
    of the operation is a collaboration with the resource managers resolving
    data flow for output Futures, which may depend on the execution context.
    """

    def __init__(self, resource_manager: SourceResource):
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
        # TODO: When the resource manager can be kept alive by an enclosing or
        #  module-level Context, convert to a weakref.
        self.__resource_manager = resource_manager
        # The unique identifier for the operation node allows the Context implementation
        # to manage the state of the handle. Reproducibility of node_uid is TBD, but
        # it must be unique in a Context where it references a different operation node.
        self.node_uid = None

    @property
    def output(self) -> DataProxyBase:
        # TODO: We can configure `output` as a data descriptor
        #  instead of a property so that we can get more information
        #  from the class attribute before creating an instance of OperationDetails.OutputDataProxy.
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


class OperationPlaceholder(AbstractOperationHandle):
    """Placeholder for Operation handle during subgraph definition."""

    def __init__(self, subgraph_resource_manager):
        ...

    def run(self):
        raise exceptions.UsageError('This placeholder operation handle is not in an executable context.')

    @property
    def output(self) -> DataProxyBase:
        """Allow subgraph components to be connected without instantiating actual operations."""
        if not isinstance(current_context(), SubgraphContext):
            raise exceptions.UsageError('Invalid access to subgraph internals.')


class NodeBuilder(abc.ABC):
    """Add an operation node to be managed by a Context."""

    @abc.abstractmethod
    def build(self) -> AbstractOperationHandle:
        ...

    @abc.abstractmethod
    def add_input(self, name: str, source):
        ...

    @abc.abstractmethod
    def add_resource_factory(self, factory):
        # The factory function takes input in the form the Context will provide it
        # and produces a resource object that will be passed to the callable that
        # implements the operation.
        assert callable(factory)
        ...

    @abc.abstractmethod
    def add_operation_details(self, operation):
        # TODO: This can be decomposed into the appropriate set of factory functions
        #  as they become clear.
        assert hasattr(operation, '_input_signature_description')
        assert hasattr(operation, 'make_uid')


class Context(abc.ABC):
    """API Context.

    All gmxapi data and operations are owned by a Context instance. The Context
    manages the details of how work is run and how data is managed.

    Context implementations are not required to inherit from gmxapi.context.Context,
    but this class definition serves to specify the current Context API.
    """

    @abc.abstractmethod
    def node_builder(self, label=None) -> NodeBuilder:
        ...


# TODO: Add required components to NodeBuilder ABC:
#  * input_signature_description
#  * ability to make_uid
#  * operation_director
#  * whatever ResourceManager _actually_ needs (instead of operation_details instance)
class ModuleNodeBuilder(NodeBuilder):
    """Builder for work nodes in gmxapi.operation.ModuleContext."""

    def __init__(self, context: 'ModuleContext', label=None):
        self.context = context
        self.label = label
        self.operation_details = None
        self.sources = DataSourceCollection()

    def add_operation_details(self, operation):
        # TODO: This can be decomposed into the appropriate set of factory functions as they become clear.
        assert hasattr(operation, '_input_signature_description')
        assert hasattr(operation, 'make_uid')
        self.operation_details = operation

    def add_input(self, name, source):
        # TODO: We can move some input checking here as the data model matures.
        self.sources[name] = source

    def add_resource_factory(self, factory):
        self.resource_factory = factory

    def build(self) -> AbstractOperationHandle:
        # Check for the ability to instantiate operations.
        if self.operation_details is None:
            raise exceptions.UsageError('Missing details needed for operation node.')

        input_sink = SinkTerminal(self.operation_details._input_signature_description)
        input_sink.update(self.sources)
        edge = DataEdge(self.sources, input_sink)
        # TODO: Fingerprinting: Each operation instance has unique output based on the unique input.
        #            input_data_fingerprint = edge.fingerprint()

        # Set up output proxy.
        uid = self.operation_details.make_uid(edge)
        # TODO: ResourceManager should fetch the relevant factories from the Context
        #  instead of getting an OperationDetails instance.
        manager = ResourceManager(source=edge, operation=self.operation_details())
        self.context.work_graph[uid] = manager
        handle = OperationHandle(self.context.work_graph[uid])
        handle.node_uid = uid
        return handle


class ModuleContext(Context):
    """Context implementation for the gmxapi.operation module.

    """
    __version__ = 0

    def __init__(self):
        self.work_graph = {}
        self.operations = {}
        self.labels = {}

    def node_builder(self, label=None) -> NodeBuilder:
        """Get a builder for a new work node to add an operation in this context."""
        if label is not None:
            if label in self.labels:
                raise exceptions.ValueError('Label {} is already in use.'.format(label))
            else:
                # The builder should update the labeled node when it is done.
                self.labels[label] = None

        return ModuleNodeBuilder(context=weakref.proxy(self), label=label)


# Context stack.
__current_context = [ModuleContext()]


def current_context() -> Context:
    """Get a reference to the currently active Context.

    The imported gmxapi.context module maintains some state for the convenience
    of the scripting environment. Internally, all gmxapi activity occurs under
    the management of an API Context, explicitly or implicitly. Some actions or
    code constructs will generate separate contexts or sub-contexts. This utility
    command retrieves a reference to the currently active Context.
    """
    return __current_context[-1]


def push_context(context) -> Context:
    """Enter a sub-context by pushing a context to the global context stack.
    """
    __current_context.append(context)
    return current_context()


def pop_context() -> Context:
    """Exit the current Context by popping it from the stack."""
    return __current_context.pop()


class OperationDirector(object):
    """Direct the construction of an operation node.

    Collaboration: used by OperationDetails.operation_director, which
    will likely dispatch to different implementations depending on
    requirements of work or context.
    """

    def __init__(self, *args, operation_details, context: Context, label=None, **kwargs):
        self.operation_details = operation_details
        self.context = weakref.proxy(context)
        self.args = args
        self.kwargs = kwargs
        self.label = label

    def __call__(self):
        cls = self.operation_details
        builder = self.context.node_builder(label=self.label)
        builder.add_operation_details(cls)

        data_source_collection = cls._input_signature_description.bind(*self.args, **self.kwargs)
        for name, source in data_source_collection.items():
            builder.add_input(name, source)
        builder.add_resource_factory(cls.resource_director)
        handle = builder.build()
        return handle


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

        # Note: Allow operations to be defined entirely in template headers to facilitate
        # compile-time optimization of fused operations. Consider what distinction, if any,
        # exists between a fused operation and a more basic operation. Probably it amounts
        # to aspects related to interaction with the Context that get combined in a fused
        # operation, such as the resource director, builder, etc.
        class OperationDetails(object):
            __basename = function.__qualname__
            __last_uid = 0
            _input_signature_description = InputCollectionDescription.from_function(function)
            # TODO: Separate the class and instance logic for the runner.
            runner = wrapped_function_runner(function, output)
            output_description = runner.output_description
            output_data_proxy = define_output_data_proxy(output_description)
            publishing_data_proxy = define_publishing_data_proxy(output_description)

            # TODO: This is a Context detail.
            @classmethod
            def make_datastore(cls, ensemble_width: int):
                datastore = {}
                for name, dtype in cls.output_description.items():
                    assert isinstance(dtype, type)
                    result_description = gmx.datamodel.ResultDescription(dtype, width=ensemble_width)
                    datastore[name] = OutputData(name=name, description=result_description)
                return datastore

            @classmethod
            def make_uid(cls, input):
                """The unique identity of an operation node tags the output with respect to the input.

                TODO: We probably don't want to allow Operations to single-handedly determine their
                 own uniqueness, but they probably should participate in the determination with the Context.

                To be refined...
                """
                uid = str(cls.__basename) + str(cls.__last_uid)
                cls.__last_uid += 1
                return uid

            @classmethod
            def resource_director(self, input=None, output=None):
                """a Director factory that helps build the Session Resources for the function.

                The Session launcher provides the director with all of the resources previously
                requested/negotiated/registered by the Operation. The director uses details of
                the operation to build the resources object required by the operation runner.

                For the Python Context, the protocol is for the Context to call the
                resource_director instance method, passing input and output containers.
                """
                resources = PyFunctionRunnerResources()
                resources.update(input.kwargs)
                resources.update({'output': output})

                # TODO: Remove this hack when we can better handle Futures of Containers and Future slicing.
                for name in resources:
                    if isinstance(resources[name], (list, tuple)):
                        resources[name] = ndarray(resources[name])

                # Check data compatibility
                for name, value in resources.items():
                    if name != 'output':
                        expected = self._input_signature_description[name]
                        got = type(value)
                        if got != expected:
                            raise exceptions.TypeError('Expected {} but got {}.'.format(expected, got))
                return resources

            @classmethod
            def operation_director(cls, *args, context: Context, label=None, **kwargs):
                """Dispatching Director for adding a work node.

                A Director for input of a particular sort knows how to reconcile
                input with the requirements of the Operation and Context node builder.
                The Director (using a less flexible / more standard interface)
                builds the operation node using a node builder provided by the Context.
                """
                if not isinstance(context, Context):
                    raise exceptions.UsageError('Context instance needed for dispatch.')
                # TODO: use Context characteristics rather than isinstance checks.
                if isinstance(context, ModuleContext):
                    construct = OperationDirector(*args, operation_details=cls, context=context, label=label, **kwargs)
                    return construct()
                elif isinstance(context, SubgraphContext):
                    construct = OperationDirector(*args, operation_details=cls, context=context, label=label, **kwargs)
                    return construct()
                else:
                    raise exceptions.ApiError('Cannot dispatch operation_director for context {}'.format(context))

        # TODO: (FR4) Update annotations with gmxapi data types. E.g. return -> Future.
        @functools.wraps(function)
        def helper(*args, context=None, **kwargs):
            # Description of the Operation input (and output) occurs in the
            # decorator closure. By the time this factory is (dynamically) defined,
            # the OperationDetails and ResourceManager are well defined, but not
            # yet instantiated.
            # Inspection of the offered input occurs when this factory is called,
            # and OperationDetails, ResourceManager, and Operation are instantiated.

            # This operation factory is specialized for the default package Context.
            if context is None:
                context = current_context()
            else:
                raise exceptions.ApiError('Non-default context handling not implemented.')

            # This calls a dispatching function that may not be able to reconcile the input
            # and Context capabilities. This is the place to handle various exceptions for
            # whatever reasons this reconciliation cannot occur.
            handle = OperationDetails.operation_director(*args, context=context, label=None, **kwargs)

            # TODO: NOW: The input fingerprint describes the provided input
            # as (a) ensemble input, (b) static, (c) future. By the time the
            # operation is instantiated, the topology of the node is known.
            # When compared to the InputCollectionDescription, the data compatibility
            # can be determined.

            return handle

        # to do: The factory itself needs to be able to register a factory with the context that will be responsible for
        # the Operation handle.
        # The factories need to be able to serve as dispatchers for themselves, since an operation in one context may
        # need to be reconstituted in a different context. The dispatching factory produces a director for a Context,
        # which will register a factory with the operation in that context.

        # The factory function has a DirectorFactory. Director instances talk to a NodeBuilder for a Context to
        # get handles to new operation nodes managed by the context. Part of that process includes registering
        # a DirectorFactory with the Context.
        return helper

    return decorator


def make_operation(implementation=None, input: dict = None, output: dict = None, docstring=None) -> typing.Callable:
    """Generate a factory function for an Operation implemented in a class.

    For an Operation implemented as a function, see function_wrapper().

    For an Operation implemented as a class named MyOperation, generate a factory
    function named "my_operation" with syntax like the following, in which "input"
    and "output" are dictionaries mapping constructor arguments and object attributes
    (respectively) to Python data types.

    Example:
        my_operation = make_operation(MyOperation, input={...}, output={...})

    Can also be used as a parameterized class decorator.

        @make_operation(input={...}, output={...})
        class MyOperation(object):
            ...

    Each output accessor is guaranteed to be called at most once per operation
    instance, regardless of the number of references held or the number of times
    `result()` is called. However, where multiple outputs may or may not be
    accessed separately, the API does not specify (at this time) whether all of
    the output accessors will be called or at what time. This behavior may be
    clarified in a future revision to the API specification or may be left as an
    implementation detail for the execution context in which the results are
    accessed.

    Args:
        implementation:
        input:
        output:
        docstring:

    Returns:
    """
    # If arguments are given, but cls is None, return a partially bound wrapper
    # for use as a decorator.
    if implementation is None:
        def decorator(cls):
            return make_operation(cls, input=input, output=output, docstring=docstring)

        return decorator

    # else: Define a function to instantiate and run the functor to populate the outputs.

    # Objects produced by the factory may be serialized and deserialized. Among
    # the simplest cases of recreating operation implementations in other environments,
    # we use a protocol mapping named operations to importable code.
    if hasattr(implementation, '__module__') and hasattr(implementation, '__name__'):
        try:
            module = importlib.util.resolve_name(name=implementation.__module__,
                                                 package=None)
            spec = importlib.util.find_spec(module)
        except ValueError:
            module = None
            spec = None
    else:
        raise exceptions.ValueError(
            'Could not determine an importable module for implementation {}'.format(implementation))
    # TODO: we can try harder, such as to inspect the __file__ and other hints for something we can import.
    if spec is None:
        raise exceptions.UsageError('make_operation can only produce Operations from importable Python code.')
    else:
        name = implementation.__name__
        module = spec.loader.exec_module(module)
        assert hasattr(module, name)

    logger.info('Generating factory for {} from {}'.format(name, spec))

    # Define and dynamically execute a wrapper definition.
    source = ['def wrapper(output, {}):',
              '    obj = implementation({})',
              '    for out in output:',
              '        setattr(output, out, getattr(obj, out))',
              ''
              ]
    source[0] = source[0].format()
    source[1] = source[1].format()
    compile(source=source, filename=spec, mode='exec')

    # TODO: Use `input` kwarg
    # to define a InputCollectionDescription and generate documentation for the factory.
    # We may want to just compile() a new code object.
    def wrapper(output, **kwargs):
        obj = implementation(**kwargs)
        for out in output:
            setattr(output, out, getattr(obj, out))

    factory = function_wrapper(output=output)(wrapper)

    return factory


class GraphVariableDescriptor(object):
    def __init__(self, name: str = None, dtype=None, default=None):
        self.name = name
        self.dtype = dtype
        self.default = default
        self.state = None

    @property
    def internal_name(self):
        try:
            return '_' + self.name
        except TypeError:
            return None

    def __get__(self, instance, owner):
        if instance is None:
            # Access is through the class attribute of the owning class.
            # Allows the descriptor itself to be inspected or reconfigured after
            # class definition.
            # TODO: There is probably some usage checking that can be performed here.
            return self
        try:
            value = getattr(instance, self.internal_name)
        except AttributeError:
            value = self.default
            # Lazily initialize the instance value from the class default.
            if value is not None:
                try:
                    setattr(instance, self.internal_name, value)
                except Exception as e:
                    message = 'Could not assign default value to {} attribute of {}'.format(
                        self.internal_name,
                        instance)
                    raise exceptions.ApiError(message) from e
        return value

    # Implementation note: If we have a version of the descriptor class with no `__set__` method,
    # it is a non-data descriptor that can be overridden by a data descriptor on an instance.
    # This could be one way to handle the polymorphism associated with state changes.
    def __set__(self, instance, value):
        if instance._editing:
            # Update the internal connections defining the subgraph.
            setattr(instance, self.internal_name, value)
        else:
            raise AttributeError('{} not assignable on {}'.format(self.name, instance))


class GraphMeta(type):
    """Meta-class for gmxapi data flow graphs and subgraphs.

    Used to implement ``subgraph`` as GraphMeta.__new__(...).
    Also allows subgraphs to be defined as Python class definitions by inheriting
    from Subgraph or by using the ``metaclass=GraphMeta`` hint in the class
    statement arguments.

    The Python class creation protocol allows Subgraphs to be defined in as follows.

    See the Subgraph class documentation for customization of instances through
    the Python context manager protocol.
    """
    _prepare_keywords = ('variables',)

    # TODO: Python 3.7.2 introduces typing.OrderedDict
    # In practice, we are using collections.OrderedDict, but we should use the generic
    # ABC from the typing module to avoid being overly restrictive with type hints.
    try:
        from typing import OrderedDict
    except ImportError:
        from collections import OrderedDict

    @classmethod
    def __prepare__(mcs, name, bases, variables: OrderedDict = None, **kwargs):
        """Prepare the class namespace.

        Keyword Args:
              variables: mapping of persistent graph variables to type / default value (optional)
        """
        # Python runs this before executing the class body of Subgraph or its
        # subclasses. This is our chance to handle key word arguments given in the
        # class declaration.

        if kwargs is not None:
            for keyword in kwargs:
                raise exceptions.UsageError('Unexpected key word argument: {}'.format(keyword))

        namespace = collections.OrderedDict()

        if variables is not None:
            if isinstance(variables, collections.abc.Mapping):
                for name, value in variables.items():
                    if isinstance(value, type):
                        dtype = value
                        if hasattr(value, 'default'):
                            default = value.default
                        else:
                            default = None
                    else:
                        default = value
                        if hasattr(default, 'dtype'):
                            dtype = default.dtype
                        else:
                            dtype = type(default)
                    namespace[name] = GraphVariableDescriptor(name, default=default, dtype=dtype)
                    # Note: we are not currently using the hook used by `inspect`
                    # to annotate with the class that defined the attribute.
                    # namespace[name].__objclass__ = mcs
                    assert not hasattr(namespace[name], '__objclass__')
            else:
                raise exceptions.ValueError('"variables" must be a mapping of graph variables to types or defaults.')

        return namespace

    def __new__(cls, name, bases, namespace, **kwargs):
        for key in kwargs:
            if key not in GraphMeta._prepare_keywords:
                raise exceptions.ApiError('Unexpected class creation keyword: {}'.format(key))
        return type.__new__(cls, name, bases, namespace)

    # TODO: This is keyword argument stripping is not necessary in more recent Python versions.
    # When Python minimum required version is increased, check if we can remove this.
    def __init__(cls, name, bases, namespace, **kwargs):
        for key in kwargs:
            if key not in GraphMeta._prepare_keywords:
                raise exceptions.ApiError('Unexpected class initialization keyword: {}'.format(key))
        super().__init__(name, bases, namespace)


class SubgraphNodeBuilder(NodeBuilder):
    def __init__(self, context: 'SubgraphContext', label=None):
        self.context = context
        self.label = label
        self.operation_details = None
        # self.input_sink = SinkTerminal(operation._input_signature_description)
        self.sources = DataSourceCollection()

    def add_resource_factory(self, factory):
        self.factory = factory

    def add_operation_details(self, operation):
        self.operation_details = operation

    def add_input(self, name: str, source):
        # Inspect inputs.
        #  * Are they from outside the subgraph?
        #  * Subgraph variables?
        #  * Subgraph internal nodes?
        # Inputs from outside the subgraph are (provisionally) subgraph inputs.
        # Inputs that are subgraph variables or subgraph internal nodes mark operations that will need to be re-run.
        # For first implementation, let all operations be recreated, but we need to
        # manage the right input proxies.
        # For zeroeth implementation, try just tracking the entities that need a reset() method called.
        if hasattr(source, 'reset'):
            self.context.add_resetter(source.reset)
        elif hasattr(source, '_reset'):
            self.context.add_resetter(source._reset)
        self.sources[name] = source

    def build(self) -> OperationPlaceholder:
        """Get a reference to the internal node in the subgraph definition.

        In the SubgraphContext, these handles cannot represent uniquely identifiable
        results. They are placeholders for relative positions in graphs that are
        not defined until the the subgraph is being executed.

        Such references should be tracked and invalidated when exiting the
        subgraph context. Within the subgraph context, they are used to establish
        the recipe for updating the subgraph's outputs or persistent data during
        execution.
        """
        # Placeholder handles in the subgraph definition don't have real resource managers.
        # Check for the ability to instantiate operations.
        if self.operation_details is None:
            raise exceptions.UsageError('Missing details needed for operation node.')

        input_sink = SinkTerminal(self.operation_details._input_signature_description)
        input_sink.update(self.sources)
        edge = DataEdge(self.sources, input_sink)
        # TODO: Fingerprinting: Each operation instance has unique output based on the unique input.
        #            input_data_fingerprint = edge.fingerprint()

        # Set up output proxy.
        uid = self.operation_details.make_uid(edge)
        # TODO: ResourceManager should fetch the relevant factories from the Context
        #  instead of getting an OperationDetails instance.
        manager = ResourceManager(source=edge, operation=self.operation_details())
        self.context.work_graph[uid] = manager
        handle = OperationHandle(self.context.work_graph[uid])
        handle.node_uid = uid
        # handle = OperationPlaceholder()
        return handle


class SubgraphContext(Context):
    """Provide a Python context manager in which to set up a graph of operations.

    Allows operations to be configured without adding nodes to the global execution
    context.
    """

    def __init__(self):
        self.labels = {}
        self.resetters = set()
        self.work_graph = collections.OrderedDict()

    def node_builder(self, label=None) -> NodeBuilder:
        if label is not None:
            if label in self.labels:
                raise exceptions.ValueError('Label {} is already in use.'.format(label))
            else:
                # The builder should update the labeled node when it is done.
                self.labels[label] = None

        return SubgraphNodeBuilder(context=weakref.proxy(self), label=label)

    def add_resetter(self, function):
        assert callable(function)
        self.resetters.add(function)


class Subgraph(object, metaclass=GraphMeta):
    """

    When subclassing from Subgraph, aspects of the subgraph's data ports can be
    specified with keyword arguments in the class statement. Example::

        >>> class MySubgraph(Subgraph, variables={'int_with_default': 1, 'boolData': bool}): pass
        ...

    Keyword Args:
        variables: Mapping to declare the types of variables with optional default values.

    Execution model:
        Subgraph execution must follow a well-defined protocol in order to sensibly
        resolve data Futures at predictable points. Note that subgraphs act as operations
        that can be automatically redefined at run time (in limited cases), such as
        to automatically form iteration chains to implement "while" loops. We refer
        to one copy / iteration / generation of a subgraph as an "element" below.

        When a subgraph element begins execution, each of its variables with an
        "updated" state from the previous iteration has the "updated" state moved
        to the new element's "initial" state, and the "updated" state is voided.

        Subgraph.next() appends an element of the subgraph to the chain. Subsequent
        calls to Subgraph.run() bring the new outputs up to date (and then call next()).
        Thus, for a subgraph with an output called ``is_converged``, calling
        ``while (not subgraph.is_converged): subgraph.run()`` has the desired effect,
        but is likely suboptimal for execution. Instead use the gmxapi while_loop.

        If the subgraph is not currently executing, it can be in one of two states:
        "editing" or "ready". In the "ready" state, class and instance Variables
        cannot be assigned, but can be read through the data descriptors. In this
        state, the descriptors only have a single state.

        If "editing," variables on the class object can be assigned to update the
        data flow defined for the subgraph. While "editing", reading or writing
        instance Variables is undefined behavior.

        When "editing" begins, each variable is readable as a proxy to the "initial"
        state in an element. Assignments while "editing" put variables in temporary
        states accessible only during editing.
        When "editing" finishes, the "updated" output data source of each
        variable for the element is set, if appropriate.

        # TODO: Use a get_context to allow operation factories or accessors to mark
        #  references for update/annotation when exiting the 'with' block.
    """


class SubgraphBuilder(object):
    """Helper for defining new Subgraphs.

    Manages a Python context in which to define a new Subgraph type. Can be used
    in a Python ``with`` construct exactly once to provide the Subgraph class body.
    When the ``with`` block is exited (or ``build()`` is called explicitly), a
    new type instance becomes available. Subsequent calls to SubgraphBuilder.__call__(self, ...)
    are dispatched to the Subgraph constructor.

    Outside of the ``with`` block, read access to data members is proxied to
    descriptors on the built Subgraph.

    Instances of SubgraphBuilder are returned by the ``subgraph()`` utility function.
    """

    def __init__(self, variables):
        self.__dict__.update({'variables': variables,
                              '_staging': collections.OrderedDict(),
                              '_editing': False,
                              '_subgraph_context': None,
                              '_subgraph_instance': None,
                              '_fused_operation': None,
                              '_factory': None})
        # Return a placeholder that we can update during iteration.
        # Long term, this is probably implemented with data descriptors
        # that will be moved to a new Subgraph type object.
        for name in self.variables:
            if not isinstance(self.variables[name], Future):
                self.variables[name] = gmx.make_constant(self.variables[name])

        # class MySubgraph(Subgraph, variables=variables):
        #     pass
        #
        # self._subgraph_instance = MySubgraph()

    def __getattr__(self, item):
        if self._editing:
            if item in self.variables:
                if item in self._staging:
                    logger.debug('Read access to intermediate value of subgraph variable {}'.format(item))
                    return self._staging[item]
                else:
                    logger.debug('Read access to subgraph variable {}'.format(item))
                    return self.variables[item]
            else:
                raise AttributeError('Invalid attribute: {}'.format(item))
        else:
            # TODO: this is not quite the described interface...
            return lambda obj: obj.values[item]

    def __setattr__(self, key, value):
        """Part of the builder interface."""
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            if self._editing:
                self.add_update(key, value)
            else:
                raise exceptions.UsageError('Subgraph is not in an editable state.')

    def add_update(self, key, value):
        """Add a variable update to the internal subgraph."""
        if key not in self.variables:
            raise AttributeError('No such attribute: {}'.format(key))
        if not self._editing:
            raise exceptions.UsageError('Subgraph is not in an editable state.')
        # Else, stage the potential final value for the iteration.
        logger.debug('Staging subgraph update {} = {}'.format(key, value))
        # Return a placeholder that we can update during iteration.
        # Long term, this is probably implemented with data descriptors
        # that will be moved to a new Subgraph type object.
        if not isinstance(value, Future):
            value = gmx.make_constant(value)
        self._staging[key] = value

    def __enter__(self):
        """Enter a Context managed by the subgraph to capture operation additions.

        Allows the internal data flow of the subgraph to be defined in the same
        manner as the default work graph while the Python context manager is active.

        The subgraph Context is activated when entering a ``with`` block and
        finalized at the end of the block.
        """
        # While editing the subgraph in the SubgraphContext, we need __get__ and __set__
        # data descriptor access on the Subgraph subclass type, but outside of the
        # context manager, the descriptor should be non-writable and more opaque,
        # while the instance should be readable, but not writable.
        self.__dict__['_editing'] = True
        # TODO: this probably needs to be configured with variables...
        self.__dict__['_subgraph_context'] = SubgraphContext()
        push_context(self._subgraph_context)
        return self

    def build(self):
        """Build the subgraph by defining some new operations.

        Examine the subgraph variables. Variables with handles to data sources
        in the SubgraphContext represent work that needs to be executed on
        a subgraph execution or iteration. Variables with handles to data sources
        outside of the subgraph represent work that needs to be executed once
        on only the first iteration to initialize the subgraph.

        Construct a factory for the fused operation that performs the work to
        update the variables on a single iteration.

        Construct a factory for the fused operation that performs the work to
        update the variables on subsequent iterations and which will be fed
        the outputs of a previous iteration. Both of the generated operations
        have the same output signature.

        Construct and wrap the generator function to recursively append work
        to the graph and update until condition is satisfied.

        TODO: Explore how to drop work from the graph once there are no more
         references to its output, including check-point machinery.
        """
        logger.debug('Finalizing subgraph definition.')
        inputs = collections.OrderedDict()
        for key, value in self.variables.items():
            # TODO: What if we don't want to provide default values?
            inputs[key] = value

        updates = self._staging

        class Subgraph(object):
            def __init__(self, input_futures, update_sources):
                self.values = collections.OrderedDict([(key, value.result()) for key, value in input_futures.items()])
                logger.debug('subgraph initialized with {}'.format(
                    ', '.join(['{}: {}'.format(key, value) for key, value in self.values.items()])))
                self.futures = collections.OrderedDict([(key, value) for key, value in input_futures.items()])
                self.update_sources = collections.OrderedDict([(key, value) for key, value in update_sources.items()])
                logger.debug('Subgraph updates staged:')
                for update, source in self.update_sources.items():
                    logger.debug('    {} = {}'.format(update, source))

            def run(self):
                for name in self.update_sources:
                    result = self.update_sources[name].result()
                    logger.debug('Update: {} = {}'.format(name, result))
                    self.values[name] = result
                # Replace the data sources in the futures.
                for name in self.update_sources:
                    self.futures[name].resource_manager = gmx.make_constant(self.values[name]).resource_manager
                for name in self.update_sources:
                    self.update_sources[name]._reset()

        subgraph = Subgraph(inputs, updates)

        return lambda subgraph=subgraph: subgraph

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the Subgraph editing session and finalize the Subgraph build.

        After exiting, this instance forwards __call__() to a factory for an
        operation that carries out the work in the subgraph with inputs bound
        in the current context as defined by ``variables``.
        """
        self._factory = self.build()

        context = pop_context()
        assert context is self._subgraph_context
        self.__dict__['_editing'] = False
        # Returning False causes exceptions in the `with` block to be reraised.
        # Remember to switch this to return True if we want to transform or suppress
        # such an exception (we probably do).
        if exc_type is not None:
            logger.error('Got exception {} while editing subgraph {}.'.format(exc_val, self))
            logger.debug('Subgraph exception traceback: \n{}'.format(exc_tb))
        return False

    def __call__(self):
        # TODO: After build() has been called, this should dispatch to a factory
        #  that returns an OperationHandle.
        return self._factory()


def while_loop(*, operation, condition, max_iteration=10):
    """Generate and run a chain of operations such that condition evaluates True.

    Returns and operation instance that acts like a single node in the current
    work graph, but which is a proxy to the operation at the end of a dynamically generated chain
    of operations. At run time, condition is evaluated for the last element in
    the current chain. If condition evaluates False, the chain is extended and
    the next element is executed. When condition evaluates True, the object
    returned by ``while_loop`` becomes a proxy for the last element in the chain.

    Equivalent to calling operation.while(condition), where available.

    Arguments:
        operation: a callable that produces an instance of an operation when called with no arguments.
        condition: a callable that accepts an object (returned by ``operation``) that returns a boolean.

    Warning:
        The protocol by which ``while_loop`` interacts with ``operation`` and ``condition``
        is very unstable right now. Please refer to this documentation when installing new
        versions of the package.

    Protocol:
        Warning:
            This protocol will be changed before the 0.1 API is finalized.

        When called, ``while_loop`` calls ``operation`` without arguments
        and captures the return value captured as ``_operation``.
        The object produced by ``operation()`` must have a ``reset``,
        a ``run`` method, and an ``output`` attribute.

        This is inspected
        to determine the output data proxy for the operation produced by the call
        to ``while_loop``. When that operation is called, it does the equivalent of

            while(condition(self._operation)):
                self._operation.reset()
                self._operation.run()

        Then, the output data proxy of ``self`` is updated with the results from
        self._operation.output.

    """
    # In the first implementation, Subgraph is NOT and OperationHandle.
    # if not isinstance(obj, AbstractOperationHandle):
    #     raise exceptions.UsageError(
    #     '"operation" key word argument must be a callable that produces an Operation handle.')
    # outputs = {}
    # for name, descriptor in obj.output.items():
    #     outputs[name] = descriptor._dtype

    # 1. Get the initial inputs.
    # 2. Initialize the subgraph with the initial inputs.
    # 3. Run the subgraph.
    # 4. Get the outputs.
    # 5. Initialize the subgraph with the outputs.
    # 6. Go to 3 if condition is not met.

    obj = operation()
    assert hasattr(obj, 'values')
    outputs = collections.OrderedDict([(key, type(value)) for key, value in obj.values.items()])

    @function_wrapper(output=outputs)
    def run_loop(output: OutputCollectionDescription):
        iteration = 0
        obj = operation()
        logger.debug('Created object {}'.format(obj))
        logger.debug(', '.join(['{}: {}'.format(key, obj.values[key]) for key in obj.values]))
        logger.debug('Condition: {}'.format(condition(obj)))
        while (condition(obj)):
            logger.debug('Running iteration {}'.format(iteration))
            obj.run()
            logger.debug(
                ', '.join(['{}: {}'.format(key, obj.values[key]) for key in obj.values]))
            logger.debug('Condition: {}'.format(condition(obj)))
            iteration += 1
            if iteration > max_iteration:
                break
        for name in outputs:
            setattr(output, name, obj.values[name])

        return obj

    return run_loop


# def subgraph(variables=None):
#     """Declare a Subgraph factory.
#
#     Produces an object used to create a *subgraph*, a fused operation that can
#     be used as a node in a work graph or to create new operations with constructs
#     like ``while_loop``.
#
#     Variables appear as both class attributes and instance attributes.
#
#     A Subgraph can be instantiated as a single-execution operation or as a more
#     complex operation, such as with `while_loop`
#
#     The returned object is a factory containing a definition for a subclass of Subgraph.
#
#     context manager behavior
#     ------------------------
#
#     The Subgraph has Python context manager behavior. Data flow within the
#     subgraph can be configured in the region of a Python ``with`` block.
#
#     When the context manager is entered and exited, the subclass definition is replaced.
#     """
#     class UserSubgraph(Subgraph, variables=variables):
#         pass
#
#     return UserSubgraph()

def subgraph(variables=None):
    """Allow operations to be configured in a sub-context.

    The object returned functions as a Python context manager. When entering the
    context manager (the beginning of the ``with`` block), the object has an
    attribute for each of the named ``variables``. Reading from these variables
    gets a proxy for the initial value or its update from a previous loop iteration.
    At the end of the ``with`` block, any values or data flows assigned to these
    attributes become the output for an iteration.

    After leaving the ``with`` block, the variables are no longer assignable, but
    can be called as bound methods to get the current value of a variable.

    When the object is run, operations bound to the variables are ``reset`` and
    run to update the variables.
    """
    # Implementation note:
    # A Subgraph (type) has a subgraph context associated with it. The subgraph's
    # ability to capture operation additions is implemented in terms of the
    # subgraph context.
    logger.debug('Declare a new subgraph with variables {}'.format(variables))

    return SubgraphBuilder(variables)
