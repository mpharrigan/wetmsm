Tutorial
========

First, make sure you have msmbuilder (at least version 3.0) and wetmsm
installed::

    $ msmb -h
    ...
    SolventShellsFeaturizer
    SolventShellsAssigner
    SolventApplyComponents
    SolventWriteVMD

Featurize and build an MSM
--------------------------

Decide which atoms you will treat as "solvent" and which will be your
"solute" atoms. For example, to use water-oxygens and protein
alpha-carbons (resp.)::

    $ msmb AtomIndices -a -p mypdb.pdb -o solute.txt --alpha
    $ msmb AtomIndices -a -p mypdb.pdb -o solvent.txt --water

Use ``SolventShellsFeaturizer`` to featurize your trajectories. ::

    $ msmb SolventShellsFeaturizer --trjs "*.dcd" --top mypdb.pdb \
        --solute_indices solute.txt \
        --solvent_indices solvent.txt
        --n_shells 4 --shell_width 0.3 --out waterfeats

You will probably want to include conformational degrees of freedom as well
as the solvent degrees of freedom. MSMBuilder will soon (as of 12/12)
support a "feature union dataset" which will concatenate two featurizations
for you via the command line.  For now, you have to do this by hand (i.e.
with ``np.concatenate(..., axis=1)``). 
        
For visualization, we recommend fitting a tICA model::

    $ msmb tICA -i waterfeats -t tica_transform -o tica_model.jl \
        --n_components 5

Assign and visualize
--------------------

Normal tICA/MSM plots and visualization are of course possible and beyond
the scope of this document. This featurization, however, affords a slightly
different type of visualization. Since there is not a 1-to-1 correspondence
between atoms and features, we assign each solvent atom to whichever
features to which it belongs and aggregate the per-feature values we wish
to visualize. ::

    $ msmb SolventShellsAssigner --trjs traj.xtc --top top.pdb \
        --solvent_indices solvent.txt --solute_indices solute.txt \
        --n_shells 4 --shell_width 0.3 --out waterassign


In this tutorial, we will use the coefficients from the first tIC learned
in our tICA model. With the following command, we combine the coefficients
with the solvent assignments::

    $ msmb SolventApplyComponents --trjs traj.xtc --top top.pdb \
        --solvent_indices solvent.txt --solute_indices solute.txt \
        --assignments waterassign --component tica_model.jl \
        --out watercomponent

Finally, we convert the MSMBuilder data to something VMD can parse.
As a convenience, this command will also generate a tcl script that
can be ``source``-ed by VMD to load in the data. ::

    $ msmb SolventWriteVMD --trj traj.xtc --top top.pdb \
        --dataset watercomponent -o watervmd


