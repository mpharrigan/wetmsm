"""Apply spherical shells solvent fingerprint to a set of trajectories."""

__author__ = 'harrigan'

import mdtraj as md
import numpy as np
from mixtape.featurizer import Featurizer
import mcmd
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

    def featurize(self, traj):
        """Featurize a trajectory using the solvent shells metric.

        Returns
        -------
        shellcounts : np.ndarray, shape=(n_frames, n_solute, n_shells)
            Number of solvent atoms in shell_i around solute_i at frame_i
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
        shellcounts = np.zeros((traj.n_frames, self.n_solute, n_shell),
                               dtype=int)
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
                # And count the number in this shell
                shellcounts[:, i, j] = np.sum(shell_bool, axis=1)

                # Build assignments chunk
                frame_solv = np.asarray(np.where(shell_bool)).T
                solu_shell = np.zeros((len(frame_solv), 2), dtype=int)

                # TODO: Why not store solute_i here?
                # Maybe to match up with "counts"
                solu_shell[:, 0] = i
                solu_shell[:, 1] = j
                assignments_list += [np.hstack((frame_solv, solu_shell))]

        return shellcounts, np.vstack(assignments_list)


class SolventShellsComputation(mcmd.Parsable):
    """Do solvent fingerprinting on trajectories.

    :param solvent_indices_fn: Path to solvent indices file
    :param solute_indices_fn: Path to solute indices file.
    :param n_shells: Number of shells to do
    :param shell_width: Width of each shell
    :param traj_fn: Trajectory filename
    :param traj_top: Trajectory topology
    :param counts_out_fn: Save total counts for each shell here
    :param assign_out_fn: Save assignments of solvent to shells here
    """

    def __init__(self, solute_indices_fn='solute_indices.dat',
                 solvent_indices_fn='solvent_indices.dat', n_shells=5,
                 shell_width=0.2, traj_fn="", traj_top="",
                 counts_out_fn='shell_count.h5',
                 assign_out_fn='shell_assign.h5'):
        # Initialize attributes
        self.solute_indices_fn = solute_indices_fn
        self.solvent_indices_fn = solvent_indices_fn
        self.n_shells = n_shells
        self.shell_width = shell_width
        self.traj_fn = traj_fn
        self.traj_top = traj_top
        self.featurizer = None
        self.feat_assn = None
        self.feat_counts = None
        self.counts_out_fn = counts_out_fn
        self.assign_out_fn = assign_out_fn

        # Load indices
        self.solvent_indices = np.loadtxt(self.solvent_indices_fn, dtype=int,
                                          ndmin=2)
        self.solute_indices = np.loadtxt(self.solute_indices_fn, dtype=int,
                                         ndmin=2)

        # Create featurizer
        self.featurizer = SolventShellsAssignmentFeaturizer(
            self.solute_indices,
            self.solvent_indices,
            self.n_shells,
            self.shell_width,
            periodic=True)

    def __str__(self):
        return "Shells: {traj_fn} with {n_shells} shells of width {shell_width} using {solute_indices_fn} and {solvent_indices_fn}".format(
            **self.__dict__
        )

    @property
    def n_solutes(self):
        return len(self.solute_indices)


    def featurize_all(self):
        """Featurize."""

        # TODO: Add option to not overwrite / Don't do computation if these exist

        # Use compression
        filters = tables.Filters(complevel=5, complib='zlib')

        # Set up assignments hdf5 file
        assn_h = tables.open_file(self.assign_out_fn, 'w')
        assn_ea = assn_h.create_earray(assn_h.root, 'assignments',
                                       atom=tables.UIntAtom(), shape=(0, 4),
                                       filters=filters)

        # Set up counts hdf5 file
        counts_h = tables.open_file(self.counts_out_fn, 'w')
        counts_ea = counts_h.create_earray(counts_h.root, 'shell_counts',
                                           atom=tables.UIntAtom(),
                                           shape=(
                                               0, self.n_solutes,
                                               self.n_shells),
                                           filters=filters)

        # Save in chunks
        for chunk in md.iterload(self.traj_fn, top=self.traj_top, chunk=5000):
            count_chunk, assn_chunk = self.featurizer.featurize(chunk)
            assn_ea.append(assn_chunk)
            counts_ea.append(count_chunk)

        # Close files
        assn_h.close()
        counts_h.close()

    def main(self):
        """Main entry point for this script."""
        self.featurize_all()


def parse():
    """Parse command line options."""
    gsc = mcmd.parsify(SolventShellsComputation)
    gsc.main()


if __name__ == "__main__":
    parse()
