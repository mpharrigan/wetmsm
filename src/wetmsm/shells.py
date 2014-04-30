"""Apply spherical shells solvent fingerprint to a set of trajectories."""

__author__ = 'harrigan'

import mdtraj as md
import numpy as np
from mixtape.featurizer import Featurizer
from mcmd import mcmd


class SolventShellsFeaturizer(Featurizer):
    def __init__(self, solute_indices, solvent_indices, n_shells, shell_width,
                 periodic=True):
        self.solute_indices = solute_indices[:, 0]
        self.solvent_indices = solvent_indices[:, 0]
        self.n_shells = n_shells
        self.shell_width = shell_width
        self.periodic = periodic
        self.n_solute = len(self.solute_indices)
        self.n_features = self.n_solute * self.n_shells

    def featurize(self, traj):
        n_shell = self.n_shells
        shell_w = self.shell_width
        shell_edges = np.linspace(0, shell_w * (n_shell + 1),
                                  num=(n_shell + 1), endpoint=True)
        shellcounts = np.zeros((traj.n_frames, self.n_solute, n_shell))
        atom_pairs = np.zeros((len(self.solvent_indices), 2))

        for i, solute_i in enumerate(self.solute_indices):
            # For each solute atom, calculate distance to all solvent
            # molecules
            atom_pairs[:, 0] = solute_i
            atom_pairs[:, 1] = self.solvent_indices

            distances = md.compute_distances(traj, atom_pairs, periodic=True)
            for j, fdist in enumerate(distances):
                hist, _ = np.histogram(fdist, bins=shell_edges)
                shellcounts[j, i, :] = hist

        return shellcounts


class SolventShellsAssignmentFeaturizer(Featurizer):
    def __init__(self, solute_indices, solvent_indices, n_shells, shell_width,
                 periodic=True):
        self.solute_indices = solute_indices[:, 0]
        self.solvent_indices = solvent_indices[:, 0]
        self.n_shells = n_shells
        self.shell_width = shell_width
        self.periodic = periodic
        self.n_solute = len(self.solute_indices)
        self.n_features = self.n_solute * self.n_shells

    def featurize(self, traj):
        n_shell = self.n_shells
        shell_w = self.shell_width
        shell_edges = np.linspace(0, shell_w * (n_shell + 1),
                                  num=(n_shell + 1), endpoint=True)

        atom_pairs = np.zeros((len(self.solvent_indices), 2))
        assignments = np.zeros(
            (traj.n_frames, len(self.solvent_indices), self.n_solute, n_shell),
            dtype=bool)
        shellcounts = np.zeros((traj.n_frames, self.n_solute, n_shell))

        for i, solute_i in enumerate(self.solute_indices):
            # For each solute atom, calculate distance to all solvent
            # molecules
            atom_pairs[:, 0] = solute_i
            atom_pairs[:, 1] = self.solvent_indices

            distances = md.compute_distances(traj, atom_pairs, periodic=True)
            for j in xrange(n_shell):
                shell_bool = np.logical_and(
                    distances >= shell_edges[j],
                    distances < shell_edges[j + 1]
                )
                assignments[:, :, i, j] = shell_bool
                shellcounts[:, i, j] = np.sum(shell_bool, axis=1)

        return assignments, shellcounts


class SolventShellsComputation(object):
    """Do solvent fingerprinting on trajectories.

    :attr solvent_indices_fn: Path to solvent indices file
    :attr solute_indices_fn: Path to solute indices file.
    """
    solute_indices = None
    solute_indices_fn = 'solute_indices.dat'
    solvent_indices = None
    solvent_indices_fn = 'solvent_indices.dat'
    n_shells = 3
    shell_width = 0.3
    traj_fn = str
    traj_top = str
    featurizer = None
    feat_assn = None
    feat_counts = None
    counts_out_fn = 'shell_count.npy'
    assign_out_fn = 'shell_assign.npy'
    trajs = None

    def load(self):
        """Load relevant data and create a featurizer object.
        """
        self.solvent_indices = np.loadtxt(self.solvent_indices_fn, dtype=int,
                                          ndmin=2)
        self.solute_indices = np.loadtxt(self.solute_indices_fn, dtype=int,
                                         ndmin=2)

        self.trajs = [md.load(self.traj_fn, top=self.traj_top)]

        self.featurizer = SolventShellsAssignmentFeaturizer(
            self.solute_indices,
            self.solvent_indices,
            self.n_shells,
            self.shell_width,
            periodic=True)

    def featurize_all(self):
        """Featurize."""
        # Note: Later will add support for multiple trajectories
        self.feat_assn, self.feat_counts = self.featurizer.featurize(
            self.trajs[0])

    def save_features(self):
        """Save solvent fingerprints to a numpy array."""
        with open(self.counts_out_fn, 'w') as f:
            np.save(f, self.feat_counts)
        with open(self.assign_out_fn, 'w') as f:
            np.save(f, self.feat_assn)

    def main(self):
        """Main entry point for this script."""
        self.load()
        self.featurize_all()
        self.save_features()


def parse():
    """Parse command line options."""
    gsc = mcmd.parsify(SolventShellsComputation)
    gsc.main()


if __name__ == "__main__":
    parse()
