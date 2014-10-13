Wetmsm
======

Investigate water in MD trajectories.

Install
-------

Install using `python setup.py install`

This package was written and tested on Python 3.4. Python 2 is supported
through the python-future package. Let me know if you run into any issues.


Using with mixtape
------------------

`wetmsm.SolventShellsFeaturizer` is compatible with mixtape's `Featurizer`
classes. Use it as you would any other featurizer. For example, with
`mixtape.featurizer.featurize_all`. This is the recommended way of using
this code. It will automatically normalize and shape the feature vectors.


Using command line scripts
--------------------------

Follow these steps to save assignments of solvent atoms to the shells
so they can be visualized later. Naively storing these assignments as a
(n_frames, n_solute, n_shells, n_solvent) array would use way too much
memory, so we write out a compressed h5 file in chunks. This requires
some special-purpose plumbing in the code.


 1. The user-facing script is `wetmsm/make_jobs.py` although I don't have
    it set to install into your path. So run `python -m wetmsm.make_jobs -h`.
    The `-h` option will show you required arguments such as topology file,
    solvent indices filename, etc. This file generates one bash script per
    trajectory so that the actual calculation can be run in parallel.

 1. Use `python -m wetmsm.make_jobs [..various options..] shells -h` to see
    solvent-shell specific parameters: shell width and number of shells

 1. You should now have one bash script per trajectory. I use GNU's `parallel`
    to run them all in parallel. You can also submit each one as a 1-core
    PBS job

 1. These calculations will produce two files per trajectory: 
        - `traj_fn.shells.h5`: (Unnormalized) features over time.
        - `traj_fn.assign.h5`: Assignment of solvent atoms to shells for
                               visualization

 1. `analysis.stepwise_analysis` takes a list of the "count" filenames,
    normalizes, removes zero-variance features, and does tICA. It saves
    the result of each as a pickle file.
