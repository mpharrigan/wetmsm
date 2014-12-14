Wetmsm
======

Investigate water in MD trajectories.

Install
-------

Install using `python setup.py install`

This package was primarily developed on python 3 and uses the python-future
package to support Python 2.  We run CI tests against python 2, but let me
know if you run into any issues.


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
                              
 - `SolventWriteVMD`:         This will export data from the above command
                              in a format readable by VMD. This command
                              will also generate a script which loads
                              the data into the `user` field in VMD. This
                              allows, e.g., coloring and selection based
                              on solvent tICA coefficients.
                        
Method Overview
---------------
![Shells schematic](/doc/source/_static/shell_fig_clip.png)

"Solvent" atoms are binned into shells surrounding "solute" atoms. The
count of atoms in each shells can be used as a feature vector for use
with MSMBuilder. By assigning solvent atoms to specific features, by
identifying important features we can actually identify individual,
relevant solvent atoms. For example, in the figure above, the bright-red
"solvent" atom is contained within two shells (features) identified as
important (perhaps by tICA). Using the provided scripts, one can
easily perform such a visualization in VMD. 
