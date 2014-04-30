"""Apply spherical shells solvent fingerprint to a set of trajectories."""

__author__ = 'harrigan'

import mdtraj as md
import numpy as np
import mixtape.featurizer
from mcmd import mcmd


class SolventShellsComputation(object):
    """Do solvent fingerprinting on trajectories.

    :attr solvent_indices_fn: Path to solvent indices file
    :attr solute_indices_fn: Path to solute indices file.
    """
    solute_indices = None
    solute_indices_fn = 'solute_indices.dat'
    solvent_indices = None
    solvent_indices_fn = 'solvent_indices.dat'
    n_shells = 6
    shell_width = 0.5
    traj_fn = str
    traj_top = str
    featurizer = None
    feat_mat = None
    feature_out_fn = 'shells.npy'
    trajs = None

    def load(self):
        """Load relevant data and create a featurizer object.
        """
        self.solvent_indices = np.loadtxt(self.solvent_indices_fn, dtype=int,
                                          ndmin=2)
        self.solute_indices = np.loadtxt(self.solute_indices_fn, dtype=int,
                                         ndmin=2)

        self.trajs = [md.load(self.traj_fn, top=self.traj_top)]

        self.featurizer = mixtape.featurizer.SolventShellsFeaturizer(
            self.solute_indices,
            self.solvent_indices,
            self.n_shells,
            self.shell_width,
            periodic=True)

    def featurize_all(self):
        """Featurize."""
        # Note: Later will add support for multiple trajectories
        self.feat_mat = self.featurizer.featurize(self.trajs[0])

    def save_features(self):
        """Save solvent fingerprints to a numpy array."""
        with open(self.feature_out_fn, 'w') as f:
            np.save(f, self.feat_mat)

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
