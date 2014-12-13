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

 - `SolventApplyComponents`:  We can leverage the linear coefficients of 
                              decomposition models like PCA or tICA
                              to enhance visualization of "important"
                              solvent atoms. This command takes as input
                              the result of `SolventShellsAssigner` (which
                              assigns solvent atoms to features) as well
                              as a serialized decomposition model
                              (PCA or tICA) and weights solvent atoms
                              by the coefficients of the features to
                              which they are assigned.
                              
 - `SolventWriteVMD`;         This will export data from the above command
                              in a format readable by VMD. This command
                              will also generate a script which loads
                              the data into the `user` field in VMD. This
                              allows, e.g., coloring and selection based
                              on solvent tICA coefficients.
                              
