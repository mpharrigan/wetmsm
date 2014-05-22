"""
Write assignments to a vmd
"""

import numpy as np
import mdtraj as md
import mcmd
import tables

VMDSCRIPT = """
mol new {top_fn} waitfor all
set mol [mol addfile {traj_fn} waitfor all]

set sel [atomselect $mol all]
set nf [molinfo $mol get numframes]
set fp [open {dat_fn} r]
set line ""

for {{set i 0}} {{$i < $nf}} {{incr i}} {{
  gets $fp line
  $sel frame $i
  $sel set user $line
}}

close $fp
$sel delete

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


class VMDWriter(object):
    def __init__(self, assn, solvent_ind, n_frames, n_solute, n_solvent, n_shells):
        self.assn = assn
        self.solvent_ind = solvent_ind

        self.to3d = None
        self.to2d = None


    def compute(self, data, features_to_select):
        """Compute loadings on each solvent atom for each frame

        :param data: 1d array of feature loadings
        :param features_to_select: Which features to consider
        """
        assn = self.assn

        for fr in range(self.n_frames):
            assn1 = assn[np.where(assn[:, 0] == fr)[0], ...]
            towrite = np.zeros(self.n_atoms)

            # Loop over features
            for feati in features_to_select:
                featu, feats = self.to3d[feati]
                logi = np.logical_and(assn1[:, 2] == featu,
                                      assn1[:, 3] == feats)
                rows = np.where(logi)[0]

                highlight = self.solvent_ind[assn1[rows, 1]]
                towrite[highlight[:, 0]] += data[feati]

            yield towrite

    def make_translation(self, deleted):
        """Turn indices from one form to another ('2d' -- '3d')

        :param deleted: Indices of states that were pruned
        """
        to3d = {}
        to2d = {}

        absi = 0  # Absolute index
        pruni = 0  # Pruned index
        for ute in range(self.n_solute):
            for sh in range(self.n_shell):
                if not np.in1d(absi, deleted):
                    to3d[pruni] = (ute, sh)
                    to2d[(ute, sh)] = pruni
                    pruni += 1
                else:
                    to2d[(ute, sh)] = -1
                absi += 1

        self.to3d = to3d
        self.to2d = to2d
        return to3d, to2d


class VMDWriterCommand(mcmd.Parsable):
    def __init__(self):
        pass


    def main(self):
        assn_h = tables.open_file(self.assn_fn)
        assn = assn_h.root.assignments

        solute_ind = None
        solvent_ind = None
