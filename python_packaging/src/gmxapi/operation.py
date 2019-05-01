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
           'concatenate_lists',
           'function_wrapper',
           'gather',
           'join_arrays',
           'make_constant',
           'scatter'
           ]

import abc
import functools
import inspect
import weakref
from contextlib import contextmanager
from typing import TypeVar

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


@computed_result
def join_arrays(a: NDArray = (), b: NDArray = ()) -> NDArray:
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
    assert isinstance(a, NDArray)
    assert isinstance(b, NDArray)
    new_list = list(a.values)
    new_list.extend(b.values)
    return new_list


Scalar = TypeVar('Scalar')


def concatenate_lists(sublists: list = ()):
    """Combine data sources into a single list.

    A trivial data flow restructuring operation
    """
    if isinstance(sublists, (str, bytes)):
        raise exceptions.ValueError('Input must be a list of lists.')
    if len(sublists) == 0:
        return ndarray([])
    else:
        return join_arrays(a=sublists[0], b=concatenate_lists(sublists[1:]))


def make_constant(value: Scalar):
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
        input_kwargs = {}
        # Note: we have also been allowing arguments with a `run` attribute to be used as execution dependencies, but we should probably stop that.
        # TODO: (FR4) generalize
        execution_dependencies = []
        if 'input' in kwargs:
            provided_input = kwargs.pop('input')
            if provided_input is not None:
                # Note: we have also been allowing arguments with a `run` attribute to be used as execution dependencies, but we should probably stop that.
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

    def __get__(self, instance: DataProxyBase, owner):
        if instance is None:
            # Access through class attribute of owner class
            return self
        resource_manager = instance._resource_instance
        client_id = instance._client_identifier
        if client_id is None:
            return getattr(resource_manager._data, self.name)
        else:
            return getattr(resource_manager._data, self.name)[client_id]

    def __set__(self, instance: DataProxyBase, value):
        resource_manager = instance._resource_instance
        client_id = instance._client_identifier
        resource_manager.set_result(name=self.name, value=value, member=client_id)

    def __repr__(self):
        return 'Publisher(name={}, dtype={})'.format(self.name, self.dtype.__qualname__)


def define_publishing_data_proxy(output_description):
    """Returns a class definition for a PublishingDataProxy for the provided output description."""
    # This dynamic type creation hides collaborations with things like make_datastore.
    # We should encapsulate these relationships in Context details, explicit collaborations
    # between specific operations and Contexts, and in groups of Operation definition helpers.

    # Dynamically define a type for the PublishingDataProxy using a descriptor for each attribute.
    # TODO: Encapsulate this bit of script in a metaclass definition?
    namespace = {}
    # Note: uses `output` from outer closure
    for name, dtype in output_description.items():
        namespace[name] = Publisher(name)
    namespace['__doc__'] = "Handler for write access to the `output` of an operation.\n\n" + \
                           "Acts as a sort of PublisherCollection."
    return type('PublishingDataProxy', (DataProxyBase,), namespace)


class SourceResource(abc.ABC):
    """Resource Manager for a data provider."""

    @classmethod
    def __subclasshook__(cls, C):
        """Check if a class looks like a ResourceManager for a data source."""
        if cls is SourceResource:
            if any('update_output' in B.__dict__ for B in C.__mro__) \
                    and hasattr(C, '_data'):
                return True
        return NotImplemented

    @abc.abstractmethod
    def is_done(self, name: str) -> bool:
        return False

    @abc.abstractmethod
    def get(self, name: str):
        return None

    @abc.abstractmethod
    def update_output(self):
        """Bring the _data member up to date and local."""
        pass


class ProxyResourceManager(SourceResource):
    """Act as a resource manager for data managed by another resource manager.

    Allow data transformations on the proxied resource.
    """

    def __init__(self, proxied_future, width: int, function):
        self._done = False
        self._proxied_future = proxied_future
        self._width = width
        self.name = self._proxied_future.name
        self._result = None
        self.function = function

    def is_done(self, name: str) -> bool:
        return self._done

    def get(self, name: str):
        if name != self.name:
            raise exceptions.ValueError('Request for unknown data.')
        if not self.is_done(name):
            raise exceptions.ProtocolError('Data not ready.')
        result = self.function(self._result)
        if self._width != 1:
            # TODO Fix this typing nightmare.
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


class Future(object):
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
        # TODO: refactor to something like resource_manager.is_done(self.name)
        assert self.resource_manager.is_done(self.name)
        # Return ownership of concrete data
        handle = self.resource_manager.get(self.name)
        return handle.data

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


class OutputDescriptor(object):
    """Read-only data descriptor for proxied output access.

    Knows how to get a Future from the resource manager.
    """

    def __init__(self, name, dtype):
        self.name = name
        assert isinstance(dtype, type)
        assert issubclass(dtype, (str, bool, int, float, dict, NDArray))
        self.dtype = dtype

    def __get__(self, proxy: DataProxyBase, owner):
        if proxy is None:
            # Access through class attribute of owner class
            return self
        result_description = gmx.datamodel.ResultDescription(dtype=self.dtype, width=proxy.ensemble_width)
        return proxy._resource_instance.future(name=self.name, description=result_description)


def define_output_data_proxy(output_description: OutputCollectionDescription):
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


class SinkTerminal(object):
    """Operation input end of a data edge.

    In addition to the information in an InputCollectionDescription, includes
    topological information for the Operation node (ensemble width).

    Collaborations: Required for creation of a DataEdge. Created with knowledge
    of a DataSourceCollection instance and a InputCollectionDescription.
    """

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
                resource = self._operation.PublishingDataProxy(weakref.proxy(self), ensemble_member)
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
        # type_annotation = self._data[name].dtype
        # type_caster = self._data[name].dtype
        # if type_caster == NDArray:
        #     type_caster = gmx.datamodel.ndarray
        # self._data[name] = Output(name=name,
        #                           dtype=type_annotation,
        #                           done=True,
        #                           data=type_caster(value))
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

    def data(self):
        """Get an adapter to the output resources to access results."""
        return self._operation.OutputDataProxy(self)

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

        # TODO: (FR3+) be more rigorous.
        #  This should probably also use a sort of Context-based observer pattern rather than
        #  the result() method, which is explicitly for moving data across the API boundary.
        # args = []
        # try:
        #     for arg in self._input_fingerprint.args:
        #         value = arg
        #         if hasattr(value, 'result'):
        #             value = value.result()
        #         args.append(value)
        # except Exception as E:
        #     raise exceptions.ApiError('input_fingerprint not iterating on "args" attr as expected.') from E

        # kwargs = {}
        # try:
        #     for key, value in self._input_fingerprint.items():
        #         if hasattr(value, 'run'):
        #             # TODO: Do we still have these?
        #             logger.debug('Calling run() for execution-only dependency {}.'.format(key))
        #             value.run()
        #             continue
        #
        #         if hasattr(value, 'result'):
        #             kwargs[key] = value.result()
        #         else:
        #             kwargs[key] = value
        #         if isinstance(kwargs[key], list):
        #             new_list = []
        #             for item in kwargs[key]:
        #                 if hasattr(item, 'result'):
        #                     new_list.append(item.result())
        #                 else:
        #                     new_list.append(item)
        #             kwargs[key] = new_list
        #         try:
        #             for item in kwargs[key]:
        #                 # TODO: This should not happen. Need proper tools for NDArray Futures.
        #                 # assert not hasattr(item, 'result')
        #                 if hasattr(item, 'result'):
        #                     kwargs[key][item] = item.result()
        #         except TypeError:
        #             # This is only a test for iterables
        #             pass
        # except Exception as E:
        #     raise exceptions.ApiError('input_fingerprint not iterating on "kwargs" attr as expected.') from E

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
                value_list = value.values
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


class CapturedOutputRunner(object):
    """Function runner that captures return value as output.data"""

    def __init__(self, function, output_description: OutputCollectionDescription):
        assert callable(function)
        self.function = function
        self.output_description = output_description
        self.capture_output = None

    def __call__(self, resources):
        if self.capture_output is None:
            raise exceptions.ProtocolError('Runner must have `capture_output` member assigned before calling.')
        self.capture_output(self.function(*resources.args, **resources.kwargs))


class OutputParameterRunner(object):
    """Function runner that uses output parameter to let function publish output."""

    def __init__(self, function, output_description: OutputCollectionDescription):
        assert callable(function)
        self.function = function
        self.output_description = output_description

    def __call__(self, resources):
        self.function(*resources.args, **resources.kwargs)


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


class OperationDetails(object):
    """Manage the implementation details of an operation instance.

    Implementation is a Python function with resources managed by a
    resource manager.

    An OperationDetails instance should be owned by the resource manager
    rather than being directly owned by the client through an Operation
    handle.
    """

    def __init__(self, function=None, output_description: dict = None,
                 input_description: InputCollectionDescription = None):
        self.runner = wrapped_function_runner(function, output_description)
        self.output_description = self.runner.output_description
        self._output_data_proxy = define_output_data_proxy(self.output_description)
        self._publishing_data_proxy = define_publishing_data_proxy(self.output_description)

        # Determine input details
        # TODO FR4: standardize typing
        self._input_signature_description = input_description

    def make_datastore(self, ensemble_width: int):
        datastore = {}
        for name, dtype in self.output_description.items():
            assert isinstance(dtype, type)
            result_description = gmx.datamodel.ResultDescription(dtype, width=ensemble_width)
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
        resources.kwargs.update(input.kwargs)
        if not hasattr(self.runner, 'capture_output'):
            resources.kwargs.update({'output': output})
        else:
            # Bind the runner's return value capture to the `data` member of `output`
            def capture(data):
                output.data = data

            self.runner.capture_output = capture

        # TODO: Remove this hack when we can better handle Futures of Containers and Future slicing.
        for name in resources.kwargs:
            if isinstance(resources.kwargs[name], (list, tuple)):
                resources.kwargs[name] = ndarray(resources.kwargs[name])

        # Check data compatibility
        for name, value in resources.kwargs.items():
            if name != 'output':
                expected = self._input_signature_description[name]
                got = type(value)
                if got != expected:
                    raise exceptions.TypeError('Expected {} but got {}.'.format(expected, got))
        return resources


class Operation(object):
    """Dynamically defined Operation handle.

    Define a gmxapi Operation for the functionality being wrapped by the enclosing code.

    An Operation type definition encapsulates description of allowed inputs
    of an Operation. An Operation instance represents a node in a work graph
    with uniquely fingerprinted inputs and well-defined output. The implementation
    of the operation is a collaboration with the resource managers resolving
    data flow for output Futures, which may depend on the execution context.
    """

    def __init__(self, resource_manager: ResourceManager):
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

    @property
    def output(self):
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
        input_collection_description = InputCollectionDescription.from_function(function)

        def get_resource_manager(source: DataEdge):
            """Provide a reference to a resource manager for the dynamically defined Operation.

            Initial Operation implementation must own ResourceManager. As more formal Context is
            developed, this can be changed to a weak reference. A distinction can also be developed
            between the facet of the Context-level resource manager to which the Operation has access
            and the whole of the managed resources.
            """
            return ResourceManager(
                source=source,
                operation=OperationDetails(function=function,
                                           output_description=output,
                                           input_description=input_collection_description))

        # TODO: (FR4) Update annotations with gmxapi data types. E.g. return -> Future.
        @functools.wraps(function)
        def factory(*args, **kwargs):
            # Description of the Operation input (and output) occurs in the
            # decorator closure. By the time this factory is (dynamically) defined,
            # the OperationDetails and ResourceManager are well defined, but not
            # yet instantiated.
            # Inspection of the offered input occurs when this factory is called,
            # and OperationDetails, ResourceManager, and Operation are instantiated.
            #
            # Per gmxapi.datamodel.DataEdge,
            # 1. Describe the data source(s)
            # 2. Compare to input description to determine sink shape.
            # 3. Build Edge with any implied data transformations.
            # 4. Make node input fingerprint and data source(s) available to resource manager.
            # 5. (TODO) Tag outputs with input fingerprint to allow unique identification of results.
            #
            # Return a handle to an operation bound to an appropriate resource manager
            # for the implementation details (wrapped function and provided input.

            # Define the unique identity and data flow constraints of this work graph node.
            data_source_collection = input_collection_description.bind(*args, **kwargs)
            sink = SinkTerminal(input_collection_description)
            sink.update(data_source_collection)
            edge = DataEdge(data_source_collection, sink)
            #            input_data_fingerprint = edge.fingerprint()
            #            input_data_fingerprint = input_collection_description.bind(*args, **kwargs)

            # Try to determine what 'input' is.
            # TODO: (FR5+) handling should be related to Context.
            #  The process of accepting input arguments includes resolving placement in
            #  a work graph and resolving the Context responsibilities for graph nodes.

            # TODO: (FR4) Check input types

            # TODO: Make allowed input strongly specified in the Operation definition.
            # TODO: Resolve execution dependencies at run() and make non-data
            #  execution `dependencies` just another input that takes the default
            #  output of an operation and doesn't do anything with it.

            # TODO: NOW: This is the place to determine whether data implies an ensemble
            #  topology or is consistent with the expected ensemble topology.

            # TODO: NOW: The input fingerprint describes the provided input
            # as (a) ensemble input, (b) static, (c) future. By the time the
            # operation is instantiated, the topology of the node is known.
            # When compared to the InputCollectionDescription, the data compatibility
            # can be determined.

            #            resource_manager = get_resource_manager(source=edge, input_fingerprint=input_data_fingerprint)
            resource_manager = get_resource_manager(source=edge)
            operation = Operation(resource_manager)
            return operation

        return factory

    return decorator


def scatter(array: NDArray) -> EnsembleDataSource:
    """Convert array data to parallel data.

    Given data with shape (M,N), produce M parallel data sources of shape (N,).

    The intention is to produce ensemble data flows from NDArray sources.
    Currently, we only support zero and one dimensional data edge cross-sections.
    In the future, it may be clearer if `scatter()` always converts a non-ensemble
    dimension to an ensemble dimension or creates an error, but right now there
    are cases where it is best just to raise a warning.

    If provided data is a string, mapping, or scalar, there is no dimension to
    scatter from and DataShapeError is raised.

    If an ensemble dimension is already present, scatter() must be able to match
    the size of that dimension or raises a DataShapeError.
    """


def gather(data: EnsembleDataSource) -> NDArray:
    """Combines parallel data to an NDArray source.

    If the data source has an ensemble shape of (1,), result is an NDArray of
    length 1 if for a scalar data source. For a NDArray data source, the
    dimensionality of the NDArray is not increased, the original NDArray is
    produced, and gather() is a no-op.

    This may change in future versions so that gather always converts an
    ensemble dimension to an array dimension.
    """
    # TODO: Could be used as part of the clean up for join_arrays to convert a scalar Future to a 1-D list.
    # Note: gather() is implemented in terms of details of the execution Context.


@function_wrapper()
def read_tpr(tprfile: str = '') -> str:
    """Prepare simulation input pack from a TPR file."""
    return tprfile
