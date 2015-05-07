"""Apply spherical shells solvent fingerprint to a set of trajectories.

Author: Matthew Harrigan
"""
from __future__ import (absolute_import, division,
                        print_function)
from future.builtins import *
import mdtraj as md
import numpy as np
import tables
import os

from . import analysis

# MSMBuilder imports
from msmbuilder.featurizer import Featurizer
from msmbuilder.dataset import MDTrajDataset
from msmbuilder.commands.featurizer import FeaturizerCommand
from msmbuilder.utils.progressbar import ProgressBar, Percentage, Bar, ETA


class SolventShellsFeaturizer(Featurizer):
    """Featurizer based on local, instantaneous water density.

    Parameters
    ----------
    solute_indices : np.ndarray, shape=(n_solute,1)
        Indices of solute atoms
    solvent_indices : np.ndarray, shape=(n_solvent, 1)
        Indices of solvent atoms
    n_shells : int
        Number of shells to consider around each solute atom
    shell_width : float
        The width of each solvent atom
    periodic : bool
        Whether to consider a periodic system in distance calculations

    Returns
    -------
    shellcounts : np.ndarray, shape=(n_frames, n_solute, n_shells)
        Number of solvent atoms in shell_i around solute_i at frame_i

    References
    ----------
    """

    def __init__(self, solute_indices, solvent_indices, n_shells, shell_width,
                 periodic=True):
        self.solute_indices = solute_indices
        self.solvent_indices = solvent_indices
        self.n_shells = n_shells
        self.shell_width = shell_width
        self.periodic = periodic
        self.n_solute = len(self.solute_indices)
        self.n_features = self.n_solute * self.n_shells

    def partial_transform(self, traj):
        """Featurize a trajectory using the solvent shells metric.

        Returns
        -------
        shellcounts : np.ndarray, shape=(n_frames, n_solute * n_shells)
            For each frame, the instantaneous density in a shell around
            a solute. Features are grouped by solute (not shell)
        """

        # Set up parameters
        n_shell = self.n_shells
        shell_w = self.shell_width
        shell_edges = np.linspace(0, shell_w * (n_shell + 1),
                                  num=(n_shell + 1), endpoint=True)

        # Initialize arrays
        atom_pairs = np.zeros((len(self.solvent_indices), 2))
        shellcounts = np.zeros((traj.n_frames, self.n_solute, n_shell),
                               dtype=np.int)

        for i, solute_i in enumerate(self.solute_indices):
            # For each solute atom, calculate distance to all solvent
            # molecules
            atom_pairs[:, 0] = solute_i
            atom_pairs[:, 1] = self.solvent_indices

            distances = md.compute_distances(traj, atom_pairs,
                                             periodic=self.periodic)

            for j in range(n_shell):
                # For each shell, do boolean logic
                shell_bool = np.logical_and(
                    distances >= shell_edges[j],
                    distances < shell_edges[j + 1]
                )
                # And count the number in this shell
                shellcounts[:, i, j] = np.sum(shell_bool, axis=1)

        shellcounts = analysis.normalize(shellcounts, shell_w)
        shellcounts = analysis.reshape(shellcounts)
        return shellcounts


class SolventShellsAssigner(SolventShellsFeaturizer):
    """Assign solvent atoms to shells to compliment SolventShellsFeaturizer.

    Parameters
    ----------
    solute_indices : np.ndarray, shape=(n_solute,1)
        Indices of solute atoms
    solvent_indices : np.ndarray, shape=(n_solvent, 1)
        Indices of solvent atoms
    n_shells : int
        Number of shells to consider around each solute atom
    shell_width : float
        The width of each solvent atom
    periodic : bool
        Whether to consider a periodic system in distance calculations

    Returns
    -------
    assignments : np.ndarray, shape=(x, 4)
        Each row corresponds to an assignment of a solvent atom to a shell
        belonging to a solute atom at a certain frame:
            (frame_i, solvent_i, solute_i, shell_i)

    """

    def partial_transform(self, traj, frame_offset):
        """Save assignments of solvent atoms to shells.

        Returns
        -------
        assignments : np.ndarray, shape=(x, 4)
            Each row corresponds to an assignment of a solvent atom to a shell
            belonging to a solute atom at a certain frame:
            (frame_i, solvent_i, solute_i, shell_i)
        """

        # Set up parameters
        n_shell = self.n_shells
        shell_w = self.shell_width
        shell_edges = np.linspace(0, shell_w * (n_shell + 1),
                                  num=(n_shell + 1), endpoint=True)

        # Initialize arrays
        atom_pairs = np.zeros((len(self.solvent_indices), 2))
        assignments_list = []

        for i, solute_i in enumerate(self.solute_indices):
            # For each solute atom, calculate distance to all solvent
            # molecules
            atom_pairs[:, 0] = solute_i
            atom_pairs[:, 1] = self.solvent_indices

            distances = md.compute_distances(traj, atom_pairs, periodic=True)

            for j in range(n_shell):
                # For each shell, do boolean logic
                shell_bool = np.logical_and(
                    distances >= shell_edges[j],
                    distances < shell_edges[j + 1]
                )

                # Build assignments chunk
                frame_solv = np.asarray(np.where(shell_bool)).T
                frame_solv[:, 0] += frame_offset
                solu_shell = np.zeros((len(frame_solv), 2), dtype=np.int)

                # TODO: Why not store solute_i here?
                # Maybe to match up with "counts"
                solu_shell[:, 0] = i
                solu_shell[:, 1] = j
                assignments_list += [np.hstack((frame_solv, solu_shell))]

        return np.vstack(assignments_list)


class SolventShellsFeaturizerCommand(FeaturizerCommand):
    """Make a MSMBuilder command-line command from our featurizer

    This registers itself in setup.py entry_points
    """
    klass = SolventShellsFeaturizer
    _concrete = True
    _group = "SolventShells1"

    def _solvent_indices_type(self, fn):
        if fn is None:
            return None
        return np.loadtxt(fn, dtype=np.int, ndmin=1)

    def _solute_indices_type(self, fn):
        if fn is None:
            return None
        return np.loadtxt(fn, dtype=np.int, ndmin=1)


class SolventShellsAssignerCommand(SolventShellsFeaturizerCommand):
    """MSMBuilder command-line command to generate assignments."""
    klass = SolventShellsAssigner
    _concrete = True
    _group = "SolventShells2"


    def start(self):
        """This method lovingly copied from MSMBuilder.

        We need to pass an extra parameter to partial transform
        """
        if self.out is not None:
            print("Warning! Please use --transformed")
            self.transformed = self.out

        if os.path.exists(self.transformed):
            self.error('File exists: %s' % self.transformed)

        print(self.instance)
        if os.path.exists(os.path.expanduser(self.top)):
            top = os.path.expanduser(self.top)
        else:
            top = None

        input_dataset = MDTrajDataset(self.trjs, topology=top,
                                      stride=self.stride, verbose=False)
        out_dataset = input_dataset.create_derived(self.transformed, fmt='dir-npy')

        pbar = ProgressBar(
            widgets=[Percentage(), Bar(), ETA()],
            maxval=len(input_dataset)).start()
        for key in pbar(input_dataset.keys()):
            trajectory = []
            for i, chunk in enumerate(
                    input_dataset.iterload(key, chunk=self.chunk)
            ):
                # ###### Pass offset as i * self.chunk!! ###########
                trajectory.append(
                    self.instance.partial_transform(
                        chunk, frame_offset=i * self.chunk
                    )
                )
            out_dataset[key] = np.concatenate(trajectory)
            out_dataset.close()

        print("\nSaving transformed dataset to '%s'" % self.transformed)
        print("To load this dataset interactive inside an IPython")
        print("shell or notebook, run\n")
        print("  $ ipython")
        print("  >>> from msmbuilder.dataset import dataset")
        print("  >>> ds = dataset('%s')\n" % self.transformed)
