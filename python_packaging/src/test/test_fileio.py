"""Test gmx.fileio submodule"""
import os
import tempfile
import unittest

import gmxapi
import pytest
from gmxapi.fileio import TprFile
from gmxapi.fileio import read_tpr
from gmxapi.exceptions import UsageError


@pytest.mark.usefixtures('cleandir')
def test_tprfile_read_old(spc_water_box):
    tpr_filename = spc_water_box
    with pytest.raises(UsageError):
        TprFile(tpr_filename, 'x')
    with pytest.raises(UsageError):
        TprFile()
    tprfile = TprFile(tpr_filename, 'r')
    with tprfile as fh:
        cpp_object = fh._tprFileHandle
        assert cpp_object is not None
        params = cpp_object.params().extract()
        assert "nsteps" in params
        assert "foo" not in params


@pytest.mark.xfail(reason='Temporarily disabled.')
@pytest.mark.usefixtures('cleandir')
def test_tprfile_read(spc_water_box):
    tprfile = gmxapi.read_tpr(spc_water_box)
    assert hasattr(tprfile, 'output')
    assert hasattr(tprfile.output, 'parameters')
    nsteps = tprfile.output.parameters['nsteps'].result()
    assert nsteps > 0


@pytest.mark.usefixtures('cleandir')
def test_core_tprcopy_alt(spc_water_box):
    """Test gmx.core.copy_tprfile() for update of end_time.

    Set a new end time that is 5000 steps later than the original. Read dt
    from file to avoid floating point round-off errors.

    Transitively test gmx.fileio.read_tpr()
    """
    tpr_filename = spc_water_box
    additional_steps = 5000
    sim_input = read_tpr(tpr_filename)
    params = sim_input.parameters.extract()
    dt = params['dt']
    nsteps = params['nsteps']
    init_step = params['init-step']
    initial_endtime = (init_step + nsteps) * dt
    new_endtime = initial_endtime + additional_steps*dt
    _, temp_filename = tempfile.mkstemp(suffix='.tpr')
    gmxapi._gmxapi.copy_tprfile(source=tpr_filename, destination=temp_filename, end_time=new_endtime)
    tprfile = TprFile(temp_filename, 'r')
    with tprfile as fh:
        params = read_tpr(fh).parameters.extract()
        dt = params['dt']
        nsteps = params['nsteps']
        init_step = params['init-step']
        assert (init_step + nsteps) * dt == new_endtime

    os.unlink(temp_filename)


@pytest.mark.usefixtures('cleandir')
def test_write_tpr_file(spc_water_box):
    """Test gmx.fileio.write_tpr_file() using gmx.core API.
    """
    tpr_filename = spc_water_box
    additional_steps = 5000
    sim_input = read_tpr(tpr_filename)
    params = sim_input.parameters.extract()
    nsteps = params['nsteps']
    init_step = params['init-step']
    new_nsteps = init_step + additional_steps

    sim_input.parameters.set('nsteps', new_nsteps)

    _, temp_filename = tempfile.mkstemp(suffix='.tpr')
    gmxapi.fileio.write_tpr_file(temp_filename, input=sim_input)
    tprfile = TprFile(temp_filename, 'r')
    with tprfile as fh:
        params = read_tpr(fh).parameters.extract()
        dt = params['dt']
        assert params['nsteps'] != nsteps
        assert params['nsteps'] == new_nsteps

    os.unlink(temp_filename)


if __name__ == '__main__':
    unittest.main()
