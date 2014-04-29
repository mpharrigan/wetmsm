__author__ = 'harrigan'

import mdtraj as md
import numpy as np
import mixtape.featurizer
from mcmd import mcmd


class GaussianSolventComputation(object):
    """
    Do solvent fingerprinting on trajectories.


    :attr solvent_indices_fn: Path to solvent indices file
    :attr solute_indices_fn: Path to solute indices file.
    """
    solute_indices = None
    solute_indices_fn = 'solute_indices.dat'
    solvent_indices = None
    solvent_indices_fn = 'solvent_indices.dat'
    sigma = 0.5
    traj_fn = str
    traj_top = str
    featurizer = None
    feat_mat = None
    feature_out_fn = 'solventfp.npy'

    def load(self):
        """Load relevant data and create a featurizer object.
        """
        self.solvent_indices = np.loadtxt(self.solvent_indices_fn, dtype=int,
                                          ndmin=2)
        self.solute_indices = np.loadtxt(self.solute_indices_fn, dtype=int,
                                         ndmin=2)

        self.trajs = [md.load(self.traj_fn, top=self.traj_top)]

        self.featurizer = mixtape.featurizer.GaussianSolventFeaturizer(
            self.solute_indices,
            self.solvent_indices,
            self.sigma,
            periodic=True)

    def featurize_all(self):
        """Featurize."""
        # Note: Later will add support for multiple trajectories
        self.feat_mat = self.featurizer.featurize(self.traj[0])

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
    gsc = mcmd.parsify(GaussianSolventComputation)
    gsc.main()


if __name__ == "__main__":
    parse()