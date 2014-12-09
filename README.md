Wetmsm
======

Investigate water in MD trajectories.

Install
-------

Install using `python setup.py install`

This package was written and tested on Python 3.4. Python 2 is supported
through the python-future package. Let me know if you run into any issues.


Using with MSMBuilder
------------------

Setup will install new subcommands to the `msmb` command.

 - `SolventShellsFeaturizer`: Featurize trajectories. Use it like any other
                              MSMBuilder featurizer.
 - `SolventShellsAssigner`:   Used for generating "assignments" of atoms
                              to shells for visualization


