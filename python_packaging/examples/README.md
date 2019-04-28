# gmxapi-scripts

Sample scripts for running some high-level tasks including ensemble simulation in gmxapi.

These scripts illustrate the functionality to be delivered in the second quarter of 2019 under subtasks
 of 
GROMACS issue [2045](https://redmine.gromacs.org/issues/2045)
and as outlined in the 
[roadmap](https://redmine.gromacs.org/projects/gromacs/repository/revisions/master/entry/python_packaging/roadmap.rst).
Syntax and comments illustrate expected user interaction with data flow between operations in a work graph.

For implementation status, refer to Gerrit [gmxapi topic](https://gerrit.gromacs.org/q/topic:%22gmxapi%22) and the 
gmxapi 
[sandbox branch](https://github.com/kassonlab/gromacs-gmxapi/commits/kassonLabFork)

## Examples

### `run_adaptive_msm.py`

Use custom analysis code from `analysis.py` to set the initial conformations for iterations of simulation batches.
