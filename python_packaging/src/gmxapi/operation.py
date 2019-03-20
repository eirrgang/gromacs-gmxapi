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
import functools
import inspect
import weakref


class AbstractResult(abc.ABC):
    """A Result serves as a proxy to or "future" for data produced by gmxapi operations.

    Result is the subset of the gmxapi Operation interface providing the output data proxy.

    An object implementing the Result interface has a `result()` method that forces resolution
    of data dependencies, triggering any necessary computation and communication, to return
    an object of the data type represented by the Result. If the Result is a container for other
    Results, nested Results are also resolved to local concrete data.
    """

    # TODO: (FR5+) A Result should defer implementation details to the Context or parent operation.
    # Note: A Result instance should not need to hold more than a weakref, if even that, to a
    # Context instance, and no reference to an Operation instance.

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

    @property
    @abc.abstractmethod
    def dtype(self):
        """Base data type of the result.

        Used to determine compatibility with the mapped inputs of consuming operations.
        """
        # At any point in time, the resource represented by this result may be in some abstract state
        # and may by an array of data sources that will be scattered to consumers.
        return None


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
        self.__input = input
        self.__cached_value = implementation(*input.args, **input.kwargs)
        # TODO: (FR4) need a utility to resolve the base type of a value that may be a proxy object.
        self._dtype = None

    def dtype(self):
        return self._dtype

    def result(self):
        return self.__cached_value


def computed_result(function):
    """Decorate a function to get a helper that produces an object with Result behavior.
    """

    @functools.wraps(function)
    def new_function(*args, **kwargs):
        """When called, the new function produces an ImmediateResult object.

        The new function has the same signature as the original function, but can accept
        proxy objects (Result objects) for arguments if the provided proxy objects represent
        a type compatible with the original signature.

        The ImmediateResult object will be evaluated in the local Context when its `result()`
        method is called the first time.

        Calls to `result()` return the value that `function` would return when executed in
        the local context with the inputs fully resolved.
        """
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
        input_pack = sig.bind(*args, **kwargs)

        result_object = ImmediateResult(function, input_pack)
        return result_object

    return new_function


@computed_result
def concatenate_lists(sublists=()):
    """Trivial data flow restructuring operation to combine data sources into a single list."""
    # TODO: (FR3) Each sublist or sublist element could be a "future" handle; make sure input provider resolves that.
    # TODO: (FR4) Returned list should be an NDArray.
    full_list = []
    for sublist in sublists:
        if sublist is not None:
            if isinstance(sublist, (str, bytes)):
                full_list.append(sublist)
            else:
                full_list.extend(sublist)
    return full_list


@computed_result
def make_constant(value):
    """Create a source of the provided value.

    Accepts a value of any type. The object returned has a definite type.
    """
    return type(value)(value)


def function_wrapper(output=()):
    """Generate a decorator for wrapped functions with signature manipulation.

    New function accepts the same arguments, with additional arguments required by
    the API.

    The new function returns an object with an `output` attribute containing the named outputs.

    Example:
        @function_wrapper(output=['spam', 'foo'])
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

    output_names = list([name for name in output])

    # Encapsulate the description of the input data flow.
    Input = collections.namedtuple('Input', ('args', 'kwargs', 'dependencies'))

    # Encapsulate the description of a data output.
    Output = collections.namedtuple('Output', ('name'))

    class OutputCollection(object):
        # TODO: Refactor with better annotated descriptors.
        # __slots__ can be considered arcane and a to be used only for optimizations.
        __slots__ = output_names

    class Publisher(object):
        """Describe an entity that produces managed outputs dependent on managed inputs."""
        def __init__(self, input):
            self._input = Input(input.args, input.kwargs, input.dependencies)
            self._outputs = {}

        def output(self, name):
            # TODO: (FR5+) should be an implementation detail of the Context and Operation.
            # Outputs should be well defined before object creation and mutable only during session lifetime.
            if name not in self._outputs:
                self._outputs[name] = Output(name)
            return self._outputs[name]

    class ResourceManager(object):
        """Provides data publication and subscription services.

        Owns data published by operation implementation or served to consumers.
        Mediates read and write access to the managed data streams.

        The `publisher` attribute is an object that can have pre-declared resources assigned as attributes.

        The `data` attribute is an object with pre-declared resources available through
        attribute access. Reading an attribute produces a new Result proxy object for
        pre-declared data or raises an AttributeError.

        TODO: This functionality should evolve to be a facet of Context implementations.
        TODO: The publisher and data objects can be more strongly defined through interaction between the Context and clients.
        """

        # Internal interface: ResourceManager provides _results_proxy(), _results_cache(), and _publish() methods.
        # These look up (or assign) the named item and either generate a new Result proxy object,
        # return the currently cached value (or None if not cached), or publish and cache the provided value.

        # We will pass the `publisher` attribute to functions that use named outputs,
        # or capture the results of simple functions to assign to a resource named `output`.
        def __init__(self):
            self._publisher = None
            self._data = OutputCollection()

        def publisher(self, node):
            """Get a handle to a publishing resource for the referenced output.

            Note: This implementation assumes there is one ResourceManager instance per publisher,
            so we only stash the inputs and dependency information for a single set of resources.
            """
            if self._publisher is None:
                self._publisher = Publisher(node._input)
            return self._publisher

        @property
        def _dependencies(self):
            return self._publisher._input.dependencies

        @property
        def output_data_proxy(self):
            """Get a Results data proxy.

            The object returned has an attribute for each output named by the publisher.
            """
            # TODO: (FR5+) Should be an implementation detail of the context implementation.
            # The gmxapi Python package provides context implementations with ensemble management.
            # A simple operation should be able to easily get an OutputResource generator and/or
            # provide a module-specific implementation.
            instance = weakref.proxy(self)
            class OutputDataProxy(object):
                # TODO: Clean up implementation.
                #  Use metaclass to configure descriptors for named outputs at higher scope.
                # TODO: (FR3+) we want some container behavior, in addition to the attributes...
                def __getattribute__(self, item):
                    if item not in instance._publisher._outputs:
                        raise AttributeError('Attribute requested is not a defined output.')
                    else:
                        # TODO: (FR3) Use Result proxy objects.
                        return getattr(instance._data, item)

            return OutputDataProxy()

        @property
        def publishing_proxy(self):
            """Get a handle to the output publishing machinery.
            """
            # The object returned has an attribute for each output named in the publisher.
            # TODO: Attributes should only be writable through a well-defined publishing protocol.
            # TODO: writing to the named outputs should allow notifications to be generated.
            return self._data

    def decorator(function):
        @functools.wraps(function)
        def new_helper(*args, **kwargs):
            def get_resource_manager():
                """Provide a reference to a resource manager for the dynamically defined Operation.

                Initial Operation implementation must own ResourceManager. As more formal Context is
                developed, this can be changed to a weak reference. A distinction can also be developed
                between the facet of the Context-level resource manager to which the Operation has access
                and the whole of the managed resources.
                """
                return ResourceManager()

            class Operation(object):
                """Dynamically defined Operation implementation.

                Define a gmxapi Operation for the functionality being wrapped by the enclosing code.
                """

                def __init__(self, *args, **kwargs):
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
                    input_args = tuple(args)
                    input_kwargs = {key: value for key, value in kwargs.items()}
                    # TODO: (FR3) generalize
                    input_dependencies = []

                    # If present, kwargs['input'] is treated as an input "pack" providing _default_ values.
                    if 'input' in input_kwargs:
                        provided_input = input_kwargs.pop('input')
                        if provided_input is not None:
                            # Try to determine what 'input' is.
                            # TODO: (FR5+) handling should be related to Context...
                            if hasattr(provided_input, 'run'):
                                input_dependencies.append(provided_input)
                            else:
                                # Assume a parameter pack is provided.
                                for key, value in provided_input.items():
                                    if key not in input_kwargs:
                                        input_kwargs[key] = value
                    assert 'input' not in input_kwargs

                    self.__input = Input(args=input_args,
                                         kwargs=input_kwargs,
                                         dependencies=input_dependencies)
                    ##

                    # TODO: (FR5+) Split the definition of the resource structure and the resource initialization.
                    # Resource structure definition logic can be moved to the level of the class definition.
                    # We need knowledge of the inputs to uniquely identify the resources for this operation instance.
                    # Implementation suggestion: Context-provided metaclass defines resource manager
                    # interface for this Operation. Factory function initializes compartmentalized
                    # resource management at object creation.
                    self.__resource_manager = get_resource_manager()
                    for name in output_names:
                        # All outputs are assumed to depend on all inputs, so we don't need to provide
                        # anything more specific than this object. The interface with the resource manager
                        # is a separate (lower level) detail.
                        node = self.__resource_manager.publisher(self)
                        node.output(name)

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

                    # TODO: Allow both structured and singular output.
                    #  Either return self._resource_manager.data or self._resource_manager.data.output
                    return self.__resource_manager.output_data_proxy

                # TODO: (FR5+) This should be composed with help from the Context implementation.
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
                    # TODO: (FR3) take action only if outputs are not already done.
                    # TODO: (FR3) make sure this gets run if outputs need to be satisfied for `result()`
                    for dependency in self.__resource_manager._dependencies:
                        dependency.run()
                    args = []
                    for arg in self.__resource_manager._publisher._input.args:
                        # TODO: (FR3+) be more rigorous...
                        if hasattr(arg, 'result'):
                            args.append(arg.result())
                        else:
                            args.append(arg)
                    kwargs = {}
                    for key, value in self.__resource_manager._publisher._input.kwargs.items():
                        if hasattr(value, 'result'):
                            kwargs[key] = value.result()
                        else:
                            kwargs[key] = value

                    assert 'input' not in kwargs
                    # TODO: Allow both structured and singular output.
                    #  For simple functions, just capture and publish the return value.
                    function(*args, output=self.__resource_manager.publishing_proxy, **kwargs)

            operation = Operation(*args, **kwargs)
            return operation

        return new_helper

    return decorator
