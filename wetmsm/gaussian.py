"""Apply Gaussian-kernel solvent fingerprint to a set of trajectories.

Author: Matthew Harrigan
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *
import mdtraj as md
import numpy as np
import mixtape.featurizer

from . import mcmd


class GaussianSolventComputation(mcmd.Parsable):
    """Do solvent fingerprinting on trajectories.


    :attr solvent_indices_fn: Path to solvent indices file
    :attr solute_indices_fn: Path to solute indices file.
    """

    def __init__(self, solute_indices_fn='solute_indices.dat',
                 solvent_indices_fn='solvent_indices.dat',
                 sigma=0.5, traj_fn='', traj_top='',
                 feature_out_fn='solventfp.npy'):
        self.solute_indices = None
        self.solute_indices_fn = solute_indices_fn
        self.solvent_indices = None
        self.solvent_indices_fn = solvent_indices_fn
        self.sigma = sigma
        self.traj_fn = traj_fn
        self.traj_top = traj_top
        self.featurizer = None
        self.feat_mat = None
        self.feature_out_fn = feature_out_fn
        self.trajs = None

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
    gsc = mcmd.parsify(GaussianSolventComputation)
    gsc.main()


if __name__ == "__main__":
    parse()
