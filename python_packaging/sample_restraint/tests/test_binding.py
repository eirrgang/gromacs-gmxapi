# The myplugin module must be locatable by Python.
# If you configured CMake in the build directory ``/path/to/repo/build`` then,
# assuming you are in ``/path/to/repo``, run the tests with something like
#     PYTHONPATH=./cmake-build-debug/src/pythonmodule mpiexec -n 2 python -m mpi4py -m pytest tests/

# This test is not currently run automatically in any way. Build the module, point your PYTHONPATH at it,
# and run pytest in the tests directory.

import pytest

from tests.conftest import withmpi_only

import logging
logging.getLogger().setLevel(logging.DEBUG)
# create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s: %(message)s')
ch.setFormatter(formatter)
# add the handlers to the logger
logging.getLogger().addHandler(ch)

logger = logging.getLogger()


def test_dependencies():
    import gmx
    assert gmx
    import gmx.core
    # holder = gmx.core.get_holder()
    # gmx.core.get_name();

def test_imports():
    import myplugin
    assert myplugin
    import gmx.core
    import gmx

@pytest.mark.usefixtures("cleandir")
def test_harmonic_potential():
    import gmx
    import os
    import myplugin

    cwd = os.path.dirname(__file__)
    water = os.path.join(cwd, 'data', 'water.gro')
    import shutil
    shutil.copy(water, './')

    try:
        # use GromacsWrapper if available
        import gromacs
        import gromacs.formats
        from gromacs.tools import Solvate as solvate
        solvate(o='water.gro', box=[5,5,5])
        mdpparams = [('integrator', 'md'),
                     ('nsteps', 1000),
                     ('nstxout', 100),
                     ('nstvout', 100),
                     ('nstfout', 100),
                     ('tcoupl', 'v-rescale'),
                     ('tc-grps', 'System'),
                     ('tau-t', 1),
                     ('ref-t', 298)]
        mdp = gromacs.formats.MDP()
        for param, value in mdpparams:
            mdp[param] = value
        mdp.write('water.mdp')
        with open('input.top', 'w') as fh:
            fh.write("""#include "gromos43a1.ff/forcefield.itp"
#include "gromos43a1.ff/spc.itp"

[ system ]
; Name
spc

[ molecules ]
; Compound  #mols
SOL         4055
""")
        gromacs.grompp(f='water.mdp', c='water.gro', po='water.mdp', pp='water.top', o='water.tpr', p='input.top')
        tpr_filename = os.path.abspath('water.tpr')
    except:
        from gmx.data import tpr_filename
    print("Testing plugin potential with input file {}".format(os.path.abspath(tpr_filename)))

    md = gmx.workflow.from_tpr(tpr_filename, append_output=False)

    context = gmx.context.ParallelArrayContext(md)
    with context as session:
        session.run()

    # Create a WorkElement for the potential
    #potential = gmx.core.TestModule()
    params = {'sites': [1, 4],
              'R0': 2.0,
              'k': 10000.0}
    potential_element = gmx.workflow.WorkElement(namespace="myplugin",
                                                 operation="create_restraint",
                                                 params=params)
    # Note that we could flexibly capture accessor methods as workflow elements, too. Maybe we can
    # hide the extra Python bindings by letting myplugin.HarmonicRestraint automatically convert
    # to a WorkElement when add_dependency is called on it.
    potential_element.name = "harmonic_restraint"
    before = md.workspec.elements[md.name]
    md.add_dependency(potential_element)
    assert potential_element.name in md.workspec.elements
    assert potential_element.workspec is md.workspec
    after = md.workspec.elements[md.name]
    assert not before is after

    # Context will need to do these in __enter__
    # potential = myplugin.HarmonicRestraint()
    # potential.set_params(1, 4, 2.0, 10000.0)

    context = gmx.context.ParallelArrayContext(md)
    with context as session:
        session.run()

@pytest.mark.usefixtures("cleandir")
def test_ensemble_potential_nompi():
    """Test ensemble potential without an ensemble.

    Still requires ParallelArrayContext.
    """
    import gmx
    import os
    import myplugin

    cwd = os.path.dirname(__file__)
    water = os.path.join(cwd, 'data', 'water.gro')
    import shutil
    shutil.copy(water, './')

    # assert False

    try:
        # use GromacsWrapper if available
        import gromacs
        import gromacs.formats
        from gromacs.tools import Solvate as solvate
        solvate(o='water.gro', box=[5,5,5])
        mdpparams = [('integrator', 'md'),
                     ('nsteps', 1000),
                     ('nstxout', 100),
                     ('nstvout', 100),
                     ('nstfout', 100),
                     ('tcoupl', 'v-rescale'),
                     ('tc-grps', 'System'),
                     ('tau-t', 1),
                     ('ref-t', 298)]
        mdp = gromacs.formats.MDP()
        for param, value in mdpparams:
            mdp[param] = value
        mdp.write('water.mdp')
        with open('input.top', 'w') as fh:
            fh.write("""#include "gromos43a1.ff/forcefield.itp"
#include "gromos43a1.ff/spc.itp"

[ system ]
; Name
spc

[ molecules ]
; Compound  #mols
SOL         4055
""")
        gromacs.grompp(f='water.mdp', c='water.gro', po='water.mdp', pp='water.top', o='water.tpr', p='input.top')
        tpr_filename = os.path.abspath('water.tpr')
    except:
        from gmx.data import tpr_filename
    print("Testing plugin potential with input file {}".format(os.path.abspath(tpr_filename)))

    assert gmx.version.api_is_at_least(0,0,5)
    md = gmx.workflow.from_tpr([tpr_filename], append_output=False)

    # Create a WorkElement for the potential
    #potential = gmx.core.TestModule()
    params = {'sites': [1, 4],
              'nbins': 10,
              'binWidth': 0.1,
              'min_dist': 0.,
              'max_dist': 10.,
              'experimental': [1.]*10,
              'nsamples': 1,
              'sample_period': 0.001,
              'nwindows': 4,
              'k': 10000.,
              'sigma': 1.}
    potential = gmx.workflow.WorkElement(namespace="myplugin",
                                         operation="ensemble_restraint",
                                         params=params)
    # Note that we could flexibly capture accessor methods as workflow elements, too. Maybe we can
    # hide the extra Python bindings by letting myplugin.HarmonicRestraint automatically convert
    # to a WorkElement when add_dependency is called on it.
    potential.name = "ensemble_restraint"
    md.add_dependency(potential)

    context = gmx.context.ParallelArrayContext(md)

    with context as session:
        session.run()


@withmpi_only
@pytest.mark.usefixtures("cleandir")
def test_ensemble_potential_withmpi():
    import gmx
    import os
    import shutil
    import myplugin

    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()

    tests_dir = os.path.dirname(__file__)
    water = os.path.join(tests_dir, 'data', 'water.gro')

    rank_dir = os.path.join(os.getcwd(), str(rank))
    os.mkdir(rank_dir)

    shutil.copy(water, rank_dir)

    # In MPI, this never makes it to grompp. We should get rid of this...
    try:
        # use GromacsWrapper if available
        import gromacs
        import gromacs.formats
        from gromacs.tools import Solvate as solvate
        solvate(o=os.path.join(rank_dir, 'water.gro'), box=[5,5,5])
        mdpparams = [('integrator', 'md'),
                     ('nsteps', 1000),
                     ('nstxout', 100),
                     ('nstvout', 100),
                     ('nstfout', 100),
                     ('tcoupl', 'v-rescale'),
                     ('tc-grps', 'System'),
                     ('tau-t', 1),
                     ('ref-t', 298)]
        mdp = gromacs.formats.MDP()
        for param, value in mdpparams:
            mdp[param] = value
        mdp.write(os.path.join(rank_dir, 'water.mdp'))
        with open(os.path.join(rank_dir, 'input.top'), 'w') as fh:
            fh.write("""#include "gromos43a1.ff/forcefield.itp"
#include "gromos43a1.ff/spc.itp"

[ system ]
; Name
spc

[ molecules ]
; Compound  #mols
SOL         4055
""")
        gromacs.grompp(f=os.path.join(rank_dir, 'water.mdp'),
                       c=os.path.join(rank_dir, 'water.gro'),
                       po=os.path.join(rank_dir, 'water.mdp'),
                       pp=os.path.join(rank_dir, 'water.top'),
                       o=os.path.join(rank_dir, 'water.tpr'),
                       p=os.path.join(rank_dir, 'input.top'))
        tpr_filename = os.path.join(rank_dir, 'water.tpr')
    except:
        from gmx.data import tpr_filename
    logger.info("Testing plugin potential with input file {}".format(os.path.abspath(tpr_filename)))

    assert gmx.version.api_is_at_least(0,0,5)
    md = gmx.workflow.from_tpr([tpr_filename, tpr_filename], append_output=False)

    # Create a WorkElement for the potential
    #potential = gmx.core.TestModule()
    params = {'sites': [1, 4],
              'nbins': 10,
              'binWidth': 0.1,
              'min_dist': 0.,
              'max_dist': 10.,
              'experimental': [0.5]*10,
              'nsamples': 1,
              'sample_period': 0.001,
              'nwindows': 4,
              'k': 10000.,
              'sigma': 1.}

    potential = gmx.workflow.WorkElement(namespace="myplugin",
                                         operation="ensemble_restraint",
                                         params=params)
    # Note that we could flexibly capture accessor methods as workflow elements, too. Maybe we can
    # hide the extra Python bindings by letting myplugin.HarmonicRestraint automatically convert
    # to a WorkElement when add_dependency is called on it.
    potential.name = "ensemble_restraint"
    before = md.workspec.elements[md.name]
    md.add_dependency(potential)

    context = gmx.context.ParallelArrayContext(md)
    with context as session:
        session.run()
