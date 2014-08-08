"""
Using assignments, write data to the User field in vmd.

This involves writing a data file and a TCL script to get VMD
to load in the data file.
"""

import os
import logging

import numpy as np
import mcmd
import tables

from math import ceil

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


class VMDWriter(object):
    """Write VMD scripts to load tICA loadings into 'user' field.

    :param assn: (M,4) array 'assignments' file
        The columns are: frame, solvent, solute, shell (indices)
    :param n_frames: Number of frames. Needed to initialize the array
    :param n_atoms: Number of all atoms. Needed to initialize the array

    :param solvent_ind: Indices of solvent atoms among all the atoms
        instead of whatever indexing is used in `assn`

    :param n_solute: Number of solute atoms for translating from 2d to 3d
    :param n_shells: Number of solvent shells. This is needed so we can
        back out the correct shape of the fingerprint vector
    """

    def __init__(self, assn, solvent_ind, n_frames, n_atoms, n_solute,
                 n_shells):
        self.assn = assn
        self.solvent_ind = solvent_ind

        self.n_frames = n_frames
        self.n_solute = n_solute
        self.n_shells = n_shells
        self.n_atoms = n_atoms


    def compute(self, loading2d, stride=1, which='add', chunksize=1000000):
        """Assign loadings to atoms based on an assignments file.

        :param loading2d: 2-d loadings (from tICA/PCA) which we apply
            to relevant atoms. Use `translate_loadings` first probably

        :param deleted: Indices (in 1d) of features that were removed
            (likely due to low-variance) before performing tICA

        :param stride: Stride output for memory reasons. All rows of the
            assignment file will still be considered

        :param which: ['add', 'max', 'avg'] How to compute loadings
            - add: Sum contributions from multiple shells
            - max: Take maximum shell contribution for each solvent atom
            - avg: WIP

        :param chunksize: How many roads of assignments to read at a time
        """

        # Initialize output arrays
        user = np.zeros((ceil(self.n_frames / stride), self.n_atoms))

        # Averaging requires keeping track of occupancies
        if which == 'avg':
            occupancy = np.zeros_like(user)

            def _compute_chunk_avg_wrapped(a, b, c, d, e):
                return _compute_chunk_avg(a, b, c, d, e, occupancy)
        else:
            def _compute_chunk_avg_wrapped(a, b, c, d, e):
                return None

        func_map = {'add': _compute_chunk_add, 'max': _compute_chunk_max,
                    'avg': _compute_chunk_avg_wrapped}
        compute_chunk = func_map[which]

        # Deal with chunks of the pytables EARRAY
        n_chunks = self.assn.shape[0] // chunksize + 1

        for chunk_i in range(n_chunks):
            chunk = self.assn.read(chunksize * chunk_i,
                                   chunksize * (chunk_i + 1))
            log.debug("Chunk %d: %s", chunk_i, str(chunk.shape))
            compute_chunk(chunk, self.solvent_ind, loading2d, user, stride)
            del chunk

        return user

    def make_translation_dicts(self, deleted):
        """Turn indices from one form to another ('2d' -- '3d')

        :param deleted: Indices of states that were pruned

        :returns to3d, to2d: Dictionaries
        """
        to3d = {}
        to2d = {}

        absi = 0  # Absolute index
        pruni = 0  # Pruned index
        for ute in range(self.n_solute):
            for sh in range(self.n_shells):
                if not np.in1d(absi, deleted):
                    to3d[pruni] = (ute, sh)
                    to2d[(ute, sh)] = pruni
                    pruni += 1
                else:
                    to2d[(ute, sh)] = -1
                absi += 1

        return to3d, to2d

    def translate_loading(self, loading, deleted):
        """Take 1-dim `loading` from tICA/PCA and expand to (solute, shell)
        indexing.

        :param loading: 1-d loadings (from tICA/PCA) which we apply
            to relevant atoms

        :param deleted: Indices (in 1d) of features that were removed
            (likely due to low-variance) before performing tICA
        """
        loading2d = np.zeros((self.n_solute, self.n_shells))

        absi = 0
        pruni = 0
        for ute in range(self.n_solute):
            for sh in range(self.n_shells):
                if not np.in1d(absi, deleted):
                    loading2d[ute, sh] = loading[pruni]
                    pruni += 1
                else:
                    loading2d[ute, sh] = 0.0

                absi += 1

        return loading2d


class VMDWriterCommand(mcmd.Parsable):
    # TODO: This command is incomplete
    # Do it by-hand in IPython for now

    def __init__(self, assn_fn='assign.h5',
                 solvent_ind_fn='solvent_indices.dat',
                 pruned_fn='fp2d_deleted.pickl', loading_data='.pickl',
                 dat_out_fn='trj.dat'):
        pass


    def main(self):
        assn_h = tables.open_file(self.assn_fn)
        assn = assn_h.root.assignments

        solute_ind = None
        solvent_ind = None


def parse():
    vc = mcmd.parsify(VMDWriterCommand)
    vc.main()


if __name__ == "__main__":
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)
    parse()

