"""
Using assignments, write data to the User field in vmd.

This involves writing a data file and a TCL script to get VMD
to load in the data file.

Author: Matthew Harrigan
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *
import logging
from math import ceil
from msmbuilder import utils
from msmbuilder.base import BaseEstimator
from msmbuilder.cmdline import NumpydocClassCommand, Command, argument, exttype
from msmbuilder.dataset import dataset, MDTrajDataset
from msmbuilder.utils.progressbar import ProgressBar, Percentage, Bar, ETA

import numpy as np
import os

from ._vmd_write import _compute_chunk_add, _compute_chunk_max, \
    _compute_chunk_avg


log = logging.getLogger()

VMDSCRIPT = """
# Load in molecule
set mol [mol new {traj_fn} step {step} waitfor all]
mol addfile {top_fn} waitfor all

# Open data file
set sel [atomselect $mol all]
set nf [molinfo $mol get numframes]
set fp [open {dat_fn} r]
set line ""

# Each line of the data file corresponds to a frame
for {{set i 0}} {{$i < $nf}} {{incr i}} {{
  gets $fp line
  $sel frame $i
  $sel set user $line
}}

close $fp
$sel delete

# For convenience, set up representations as well

mol delrep 0 top

mol representation NewCartoon 0.3 10.0 4.1 0
mol color ColorID 4
mol selection {{protein}}
mol addrep top
mol smoothrep top 0 5


mol representation CPK 1.0 0.2 10.0 10.0
mol color User
mol selection {{user > 1}}
mol addrep top
mol selupdate 1 top 1
mol colupdate 1 top 1
"""


def translate_component(component, n_solute, n_shells, deleted=None):
    """Take 1d component from tICA/PCA and expand to (solute, shell) indexing.

    Parameters
    ----------
    component : array_like, shape=(n_solute * n_shells - n_deleted)
        1-d loadings (from tICA/PCA) for transformation
    n_solute : int
        Number of solute atoms
    n_shells : int
        Number of shells
    deleted : array_like, shape=(n_deleted), default=None
        Indices (1-d) of features that were removed (likely due to
        low-variance) before performing tICA.

    Returns
    -------
    component2d : np.ndarray, shape=(n_solute, n_shells)
        2-d component.
    """
    component2d = np.zeros((n_solute, n_shells))
    if deleted is None:
        deleted = np.asarray([])

    assert n_solute * n_shells - len(deleted) == len(component)

    absi = 0
    pruni = 0
    for ute in range(n_solute):
        for sh in range(n_shells):
            if not np.in1d(absi, deleted):
                component2d[ute, sh] = component[pruni]
                pruni += 1
            else:
                component2d[ute, sh] = 0.0

            absi += 1

    return component2d


class ApplyComponents(BaseEstimator):
    """Take a vector and apply it to Assigned solvent atoms.

    Supply a vector (e.g. a tICA component of interest) and this
    function will apply each vector value to each assigned solvent
    atom

    Parameters
    ----------
    component : np.ndarray, shape=(n_features)
        The values to apply to each feature. If used from the API, any
        vector can be used. If run from the command-line, we use
        msmbuilder.utils.load to load a serialized decomposition object
        and use the 0th element of the components_ attribute. To specify
        which component, separate the filename and index with a colon:
        e.g. --component my_tica.jl:2 will use components_[2].
    solvent_indices : array-like of ints
        Solvent indices
    solute_indices : array-like of ints
        Solute indices
    agg_method : {'add', 'avg', 'max'}
        Some atoms may be assigned to multiple features. This toggles
        how the final value is computed (aggregated) from the potentially
        many input values. 'add' uses the sum of all feature components,
        'avg' computes the mean, and 'max' chooses the maximum loading
        among features.
    stride : int
        Stride the output array to save memory.

    """

    def __init__(self, component, solvent_indices, solute_indices,
                 agg_method='add', stride=1):
        self.agg_method = agg_method
        self.stride = stride
        self.component = component
        self.solvent_indices = solvent_indices
        self.solute_indices = solute_indices


    def fit(self, X, y=None):
        return self

    def unpack_component(self):
        """Translate component to two dimensions"""
        if self.component.ndim == 1:
            n_solute = len(self.solute_indices)
            n_shells = len(self.component) // n_solute
            assert len(self.component) % n_solute == 0
            component2d = translate_component(self.component, n_solute,
                                              n_shells)
        else:
            component2d = self.component

        return component2d

    def partial_transform(self, traj_assn):
        # Unpack input
        traj, assn = traj_assn
        assn = np.asarray(assn, dtype='uint32')

        # Initialize output arrays
        user = np.zeros((ceil(traj.n_frames / self.stride), traj.n_atoms))

        # Averaging requires keeping track of occupancies
        if self.agg_method == 'avg':
            occupancy = np.zeros_like(user)

            def _compute_chunk_avg_wrapped(a, b, c, d, e):
                return _compute_chunk_avg(a, b, c, d, e, occupancy)
        else:
            def _compute_chunk_avg_wrapped():
                return None

        # Select the correct function
        func_map = {'add': _compute_chunk_add, 'max': _compute_chunk_max,
                    'avg': _compute_chunk_avg_wrapped}
        compute_chunk = func_map[self.agg_method]


        # Call the appropriate computation function
        component2d = self.unpack_component()
        compute_chunk(assn, self.solvent_indices, component2d, user,
                      self.stride)

        # For averaging, we have to do the division at the end
        if self.agg_method == 'avg':
            inds = occupancy > 0
            ret = np.zeros_like(user)
            ret[inds] = user[inds] / occupancy[inds]
            return ret

        return user

    def plot_component(self, cutoff=0.0, title_prefix='The'):
        """Visualize shell component weights

        Parameters
        ----------
        cutoff : float
            Remove points below this threshold
        title_prefix : str
            Each subplot will be prefixed with this. E.g. "tICA", "PCA"
        """
        # X range
        from matplotlib import pyplot as plt
        xx = np.arange(len(self.component))
        xlim1 = (0, len(self.component))
        plt.subplots(2, 2, figsize=(10, 6))

        plt.subplot(2, 2, 1)
        plt.title('{} Components'.format(title_prefix))
        plt.scatter(xx, self.component, linewidth=0, s=30, edgecolor='none')
        plt.xlabel('Feature')
        plt.xlim(xlim1)

        plt.subplot(2, 2, 2)
        plt.title('{} Component Intensities'.format(title_prefix))
        tic_sq = self.component ** 2
        tic_sq /= np.max(tic_sq)
        plt.scatter(xx, tic_sq, linewidth=0, s=30, c='r', edgecolor='none')
        plt.xlabel('Feature')
        plt.xlim(xlim1)

        plt.subplot(2, 2, 3)
        plt.title('{} Clipped Intensities'.format(title_prefix))
        loading1d = np.copy(tic_sq)
        loading1d[tic_sq < cutoff] = 0.0
        plt.scatter(xx, loading1d, linewidth=0, s=30, c='purple',
                    edgecolor='none')
        plt.xlabel('Feature')
        plt.xlim(xlim1)

        plt.subplot(2, 2, 4)
        plt.title('Loadings by shell')
        loading2d = self.unpack_component()
        xlim2 = (0, loading2d.shape[0])
        utebins, shbins = np.meshgrid(np.arange(loading2d.shape[0]),
                                      np.arange(loading2d.shape[1]))
        plt.scatter(utebins, shbins, edgecolor='none',
                    s=300 * loading2d, c=-loading2d, cmap=plt.get_cmap('RdBu'))
        plt.ylabel('Shell')
        plt.xlabel('Residue')
        plt.xlim(xlim2)

        plt.tight_layout()
        return loading1d, loading2d


class ApplyComponentsCommand(NumpydocClassCommand):
    klass = ApplyComponents
    _concrete = True
    _group = 'SolventShells3'

    @classmethod
    def _get_name(cls):
        return "SolventApplyComponents"

    trjs = argument(
        '--trjs', help='Glob pattern for trajectories',
        default='', required=True)
    top = argument(
        '--top', help='Path to topology file matching the trajectories',
        default='')
    out = argument(
        '--out', required=True, help='Output path', type=exttype('/'))
    assn = argument(
        '--assignments', help="Assignments dataset from SolventShellsAssigner",
        required=True
    )

    def _solvent_indices_type(self, fn):
        if fn is None:
            return None
        return np.loadtxt(fn, dtype=int, ndmin=1)

    def _solute_indices_type(self, fn):
        if fn is None:
            return None
        return np.loadtxt(fn, dtype=int, ndmin=1)

    def _component_type(self, spec):
        if spec is None:
            return None
        spec_split = spec.split(':')
        if len(spec_split) > 1:
            fn = ':'.join(spec_split[:-1])
            comp_i = int(spec_split[-1])
        else:
            fn = spec_split[0]
            comp_i = 0

        obj = utils.load(fn)
        component = obj.components_[comp_i]
        component = component ** 2
        component = component / np.max(component)
        return component


    def start(self):
        if os.path.exists(self.out):
            self.error('File exists: %s' % self.out)

        print(self.instance)
        if os.path.exists(os.path.expanduser(self.top)):
            top = os.path.expanduser(self.top)
        else:
            top = None

        traj_dataset = MDTrajDataset(self.trjs, topology=top,
                                     stride=self.instance.stride, verbose=False)

        with dataset(self.assn, mode='r') as assn_dataset:
            out_dataset = assn_dataset.create_derived(self.out, fmt='dir-npy')
            pbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()],
                               maxval=len(assn_dataset)).start()
            for tr_key, as_key in pbar(
                    zip(traj_dataset.keys(), assn_dataset.keys())
            ):
                out_dataset[as_key] = self.instance.partial_transform(
                    (traj_dataset[tr_key], assn_dataset[as_key])
                )
            out_dataset.close()

        print("\nSaving transformed dataset to '%s'" % self.out)
        print("To load this dataset interactive inside an IPython")
        print("shell or notebook, run\n")
        print("  $ ipython")
        print("  >>> from msmbuilder.dataset import dataset")
        print("  >>> ds = dataset('%s')\n" % self.out)


class WriteVMDCommand(Command):
    """Write out data in VMD-compatible format.

    Note that for full generality, this command will write out
    a n_frames x n_atoms plaintext file for reading by VMD. This file
    could be quite large.

    The resulting tcl script will load the molecule and then read the
    visualization data into it. To load it from the command line use

        $ vmd -e out_prefix.tcl

    To use it from an already-running instance of VMD:

        >>> source out_prefix.tcl

    If you already have your molecule loaded, delete the first two lines
    of the output script and set the $mol variable to your molecule. For
    example:

        >>> set mol top
        >>> source out_prefix.modified.tcl
    """
    _concrete = True
    _group = 'SolventShells4'
    description = __doc__

    @classmethod
    def _get_name(cls):
        return "SolventWriteVMD"

    dataset = argument("-ds", "--dataset",
             help="Path to the dataset.",
             required=True)
    out_prefix = argument('-o', '--out_prefix',
             help="Prefix output files with this")
    traj = argument('--trj',
             help="Trajectory to load in VMD",
             default="traj.dcd")
    top = argument('--top',
             help="Topology to load in VMD",
             default="top.pdb")
    stride = argument('--stride',
             help="Stride by this when loading in VMD. Match this to"
                  " whatever value you used in ApplyComponents",
             type=int, default=1)


    def __init__(self, args):
        self.dataset = args.dataset
        self.out_prefix = args.out_prefix
        self.traj = args.trj
        self.top = args.top
        self.stride = args.stride

    def start(self):
        ds = dataset(self.dataset, mode='r')
        assert len(ds) == 1, "Only support one at a time for now"
        ds = ds[0]

        dat_fn = "{}.txt".format(self.out_prefix)
        tcl_fn = "{}.tcl".format(self.out_prefix)
        np.savetxt(dat_fn, ds, fmt="%.5f")
        with open(tcl_fn, 'w') as f:
            f.write(VMDSCRIPT.format(
                traj_fn=self.traj, step=self.stride, top_fn=self.top,
                dat_fn=dat_fn
            ))



