========================
Python packaging roadmap
========================

Describe functional goals, feature tracking, and associated tests.

Functionality to build
======================

General tool implementation sequence
------------------------------------

1. gmx.make_operation() wraps Python code with compliant interface
2. gmx.commandline_operation() provides UI for (nearly) arbitrary CLI tools
3. gmx.tool module provides trivially wrapped ``gmx`` tools with well-defined inputs and outputs

Simulation tool implementation sequence
---------------------------------------

1. gmx.mdrun
2. gmx.modify_input
3. gmx.make_input

Data flow
---------

* extract() methods force data locality for non-API calls
* output properties allow proxied access to (fault tolerant, portable) result futures
* gmx.logical_and(), etc. allow manipulation of data / signals "in flight"

Data flow topology tools
------------------------

* implicit scatter / map
* implicit broadcast
* gmx.gather() makes results local
* gmx.reduce() allows operation across an ensemble (implicit allReduce where appropriate)

Control flow
------------

* gmx.subgraph allows several operations to be bundled with a scope that allows
  user-declared data persistence (e.g. when used in a loop)
* gmx.while allows repeated execution of a subgraph, subject to a declared
  condition of the subgraph state/output, managing accessibilty of data handles
  internal and external to the graph

Execution environment
---------------------

1. gmx.run() launches work to produce requested data
2. infrastructure manages / negotiates dispatching or allocation of available resources

Packaging
---------

Installation procedures should be testable. CI scripts work for this.
Ref: https://github.com/kassonlab/gmxapi/blob/master/ci_scripts/test_installers.sh

1. Python package can be installed from GROMACS source directory after GROMACS installation.
2. Python package can be installed by a user from a hand-built GROMACS installation.
3. Python package can be installed from a Linux binary GROMACS distribution using
  appropriate optimized binary libraries.

Typing
------

* NDArray supports an N-dimensional array of a single scalar gmxapi data type
  (with sufficient metadata for common array view APIs)
* Integer32 32-bit integer scalar
* Integer64 64-bit integer scalar
* Float32 32-bit floating point scalar
* Float64 64-bit floating point scalar
* Boolean logical scalar
* String portable variable-length text string
* Byte 8-bit scalar
* gmx.Map supports an associative container with key strings and Variant values

NDArray is immediately necessary to disambiguate non-scalar parameters/inputs from
implicit data flow topological operations.

Data structures currently coupled to file formats should be decomposable into
Maps with well-specified schema, but without API data abstractions we just use
filename Strings for input/output data. (Suggest: no need for file objects at
the higher-level API if read_Xfiletype(/write_Yfiletype) operations
produce(/consume) named and typed data objects of specified API types.)

Note: need policy/convention regarding implicit narrowing/widening.
* JSON does not have a BLOB data standard
* JSON spec defers character encoding for strings to the file level (Python assumes utf-8 ASCII)
* JSON integers precision is unspecified (Python uses 64-bit)
* JSON floating point precision is unspecified (Python uses 64-bit)

Sequence
========

Features development sequence based on functional priorities and dependencies.

.. extracted with
python -c \
"import json;
print('\n'.join(
    [str('* ' + '  '.join([line for line in cell['source']]))
        for cell in json.load(open('RequiredFunctionality.ipynb', 'r'))['cells']
        if cell['cell_type'] == 'markdown']
    )
)" | \
sed 's/<!-- /\*\(/' | \
sed 's/ -->/\)\*/'

* gmx.make_operation wraps importable Python code.
  *(24 January)*
* gmx.make_operation produces output proxy that establishes execution dependency
  *(1 February)*
* gmx.make_operation produces output proxy that can be used as input
  *(5 February)*
* gmx.make_operation uses dimensionality and typing of named data to generate correct work topologies
  *(8 February)*
* gmx.gather allows explicit many-to-one or many-to-many data flow
  *(15 February)*
* gmx.reduce helper simplifies expression of operations dependent on gather
  *(15 February)*
* gmx.commandline_operation provides utility for wrapping command line tools
  *(15 February)*
* gmx.commandline_operation produces operations that can be executed in a dependency graph.
  *(15 February)*
* gmx.mdrun uses bindings to C++ API to launch simulations
  *(22 February)*
* gmx.mdrun understands ensemble work
  *(22 February)*
* *gmx.mdrun supports interface for binding MD plugins*
  (requires interaction with library development)
  *(1 March)*
* gmx.subgraph fuses operations
  *(1 March)*
* gmx.while creates an operation wrapping a dynamic number of iterations of a subgraph
  *(1 March)*
* gmx.logical_* operations allow optimizable manipulation of boolean values
  *(8 March)*
* gmx.read_tpr utility provides access to TPR file contents
  *(22 February)*
* gmx.read_tpr operation produces output consumable by gmx.mdrun
  *(22 February)*
* gmx.mdrun produces gromacs.read_tpr node for tpr filename kwargs
  *(22 February)*
* gmx.mdrun is properly restartable
  *(22 February)*
* gmx.run finds and runs operations to produce expected output files
  *(8 March)*
* gmx.run handles ensemble work topologies
  *(8 March)*
* gmx.run handles multi-process execution
  *(8 March)*
* gmx.run safety checks to avoid data loss / corruption
  *(8 March)*
* *gmx.run conveys run-time parameters to execution context*
  (requires interaction with library development)
  *(15 March)*
* *gmx.modify_input produces new (tpr) simulation input in data flow operation*
  (requires interaction with library development)
  *(1 March)*
* gmx.make_input dispatches appropriate preprocessing for file or in-memory simulation input.
  *(15 March)*
* *gmx.make_input handles state from checkpoints*
  (requires interaction with library development)
  *(22 March)*
* gmx.write_tpr (a facility used to implement higher-level functionality) merges tpr data (e.g. inputrec, structure, topology) into new file(s)
  *(1 March)*
* gmx.tool provides wrapping of unmigrated gmx CLI tools
  *(1 March)*
* gmx.tool uses Python bindings on C++ API for CLI modules
  *(15 March)*
* *gmx.tool operations are migrated to updated Options infrastructure*
  (requires interaction with library development)
  *(5 April)*
* gmx.context manages data placement according to where operations run
  *(8 March)*
* *gmx.context negotiates allocation of 1 node per operation with shared comm*
  (requires interaction with library development)
  *(8 March)*
* gmx.context negotiates an integer number of nodes per operation
  *(22 March)*
* *gmx.context negotiates allocation of resources for scheduled work*
  (requires interaction with library development)
  *(19 April)*

Expectations on GROMACS master changes
======================================

* libgmxapi and libgromacs need to be combined
* gmx Variant and types would be very helpful at API boundary
* UI helpers should express
  - preferred name for datum as a string
  - setter
  - typing and type discovery
  - help text
  - (for CLI: short name for flag)
* library should be available for static linking with position independent code
  to allow a single shared-object to be built for the Python module.
