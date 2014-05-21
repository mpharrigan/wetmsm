"""
Write assignments to a vmd
"""

import numpy as np
import mdtraj as md
import mcmd


class VMDWriter(object):
    def __init__(self, n_atoms):
        self.n_atoms = n_atoms
        self.assn = None
        self.features_to_select
        self.to3d
        self.solvent_ind

    def compute(self, data):
        assn = self.assn

        for fr in range(self.n_frames):
            assn1 = assn[np.where(assn[:, 0] == fr)[0], ...]
            towrite = np.zeros(self.n_atoms)

            # Loop over features
            for feati in self.features_to_select:
                featu, feats = self.to3d[feati]
                logi = np.logical_and(assn1[:, 2] == featu,
                                      assn1[:, 3] == feats)
                rows = np.where(logi)[0]

                highlight = self.solvent_ind[assn1[rows, 1]]
                towrite[highlight[:, 0]] += data[feati]

            yield towrite


class VMDWriterCommand(mcmd.Parsable):
    # TODO: Rebase on new mcmd
    pass
