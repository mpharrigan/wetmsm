"""Apply spherical shells solvent fingerprint to a set of trajectories."""

__author__ = 'harrigan'

import mdtraj as md
import numpy as np
from mixtape.featurizer import Featurizer
from mcmd import mcmd
import tables


class SolventShellsAssignmentFeaturizer(Featurizer):
    """Bin solvent atoms into spherical shells around solute atoms.

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
    shellcounts : np.ndarray, shape=(n_frames, n_solute, n_shells)
        Number of solvent atoms in shell_i around solute_i at frame_i

    References
    ----------
    """

    def __init__(self, solute_indices, solvent_indices, n_shells, shell_width,
                 periodic=True):
        self.solute_indices = solute_indices[:, 0]
        self.solvent_indices = solvent_indices[:, 0]
        self.n_shells = n_shells
        self.shell_width = shell_width
        self.periodic = periodic
        self.n_solute = len(self.solute_indices)
        self.n_features = self.n_solute * self.n_shells

        self.shell_counts = None

    def featurize(self, traj):

        # Set up parameters
        n_shell = self.n_shells
        shell_w = self.shell_width
        shell_edges = np.linspace(0, shell_w * (n_shell + 1),
                                  num=(n_shell + 1), endpoint=True)


        # Initialize arrays
        atom_pairs = np.zeros((len(self.solvent_indices), 2))
        shellcounts = np.zeros((traj.n_frames, self.n_solute, n_shell),
                               dtype=int)

        for i, solute_i in enumerate(self.solute_indices):
            # For each solute atom, calculate distance to all solvent
            # molecules
            atom_pairs[:, 0] = solute_i
            atom_pairs[:, 1] = self.solvent_indices

            distances = md.compute_distances(traj, atom_pairs, periodic=True)

            for j in xrange(n_shell):
                # For each shell, do boolean logic
                shell_bool = np.logical_and(
                    distances >= shell_edges[j],
                    distances < shell_edges[j + 1]
                )
                # And count the number in this shell
                shellcounts[:, i, j] = np.sum(shell_bool, axis=1)

                # Build assignments chunk
                frame_solv = np.asarray(np.where(shell_bool)).T
                solu_shell = np.zeros((len(frame_solv), 2), dtype=int)
                solu_shell[:, 0] = i
                solu_shell[:, 1] = j
                assignments_chunk = np.hstack((frame_solv, solu_shell))

                yield assignments_chunk

        # Put this in an attribute
        self.shell_counts = shellcounts


class SolventShellsComputation(mcmd.Parsable):
    """Do solvent fingerprinting on trajectories.

    :attr solvent_indices_fn: Path to solvent indices file
    :attr solute_indices_fn: Path to solute indices file.
    :attr n_shells: Number of shells to do
    :attr shell_width: Width of each shell
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
    counts_out_fn = 'shell_count.h5'
    assign_out_fn = 'shell_assign.h5'
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

        # TODO: Add option to not overwrite / Don't do computation if these exist

        # Use compression
        filters = tables.Filters(complevel=5, complib='zlib')

        # Set up assignments hdf5 file
        assn_h = tables.open_file(self.assign_out_fn, 'w')
        assn_ea = assn_h.create_earray(assn_h.root, 'assignments',
                                       atom=tables.UIntAtom(), shape=(0, 4),
                                       filters=filters)

        # Save in chunks
        for assn_chunk in self.featurizer.featurize(self.trajs[0]):
            assn_ea.append(assn_chunk)

        # Close that file
        assn_h.close()

        # Save shell counts
        counts_h = tables.open_file(self.counts_out_fn, 'w')
        counts_shape = self.featurizer.shell_counts.shape
        counts_ca = counts_h.create_carray(counts_h.root, 'shell_counts',
                                           atom=tables.UIntAtom(),
                                           shape=counts_shape, filters=filters)
        counts_ca[...] = self.featurizer.shell_counts
        counts_h.close()


    def main(self):
        """Main entry point for this script."""
        self.load()
        self.featurize_all()


def parse():
    """Parse command line options."""
    gsc = mcmd.parsify(SolventShellsComputation)
    gsc.main()


if __name__ == "__main__":
    parse()
