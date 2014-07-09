"""
Using assignments, write data to the User field in vmd.
"""

import os
import logging

import numpy as np
import mcmd
import tables

#from ._vmd_write import _compute, _translate


log = logging.getLogger()

VMDSCRIPT = """
set mol [mol new {traj_fn} step {step} waitfor all]
mol addfile {top_fn} waitfor all

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


def _compute(assn, loading, to2d, n_frames, n_atoms, solvent_ind, stride):
    """Add "loading" to each relevant atom

    :param assn: (M,4) array 'assignments' file
        The columns are: frame, solvent, solute, shell (indices)

    :param loading: Values to apply to relevant features


    """
    user = np.zeros((n_frames, n_atoms), dtype=np.float)

    for i in range(assn.shape[0]):
        fr = assn[i, 0]
        vent = solvent_ind[assn[i, 1]]
        ute_shell = to2d[(assn[i, 2], assn[i, 3])]

        user[fr, vent] += loading[ute_shell]

    return user


class VMDWriter(object):
    """Write VMD scripts to load tICA loadings into 'user' field.

    :param assn: Assignment array
    :param solvent_ind: Solvent indices
    :param n_frames: Number of frames. TODO: Why do we need this?
    :param n_atoms: Number of *all* atoms. VMD needs a value for each atom
    :param n_solute: Number of solute atoms
    :param n_shells: Number of solvent shells. This is needed so we can
        back out the correct shape of the fingerprint vector
    """

    def __init__(self, assn, solvent_ind, n_frames, n_atoms, n_solute,
                 n_shells):
        self.assn = assn
        self.solvent_ind = solvent_ind

        self.to3d = None
        self.to2d = None

        self.n_frames = n_frames
        self.n_solute = n_solute
        self.n_shells = n_shells
        self.n_atoms = n_atoms

    def compute_old(self, data, features_to_select, stride=1):
        """Compute loadings on each solvent atom for each frame

        :param data: 1d array of feature loadings
        :param features_to_select: Which features to consider
        :param stride: Do every n-th frame
        """
        assn = self.assn

        for fr in range(0, self.n_frames, stride):
            assn1 = assn[np.where(assn[:, 0] == fr)[0], ...]
            towrite = np.zeros(self.n_atoms)

            # Loop over features
            for feati in features_to_select:
                featu, feats = self.to3d[feati]
                logi = np.logical_and(assn1[:, 2] == featu,
                                      assn1[:, 3] == feats)
                rows = np.where(logi)[0]

                highlight = self.solvent_ind[assn1[rows, 1]]
                towrite[highlight[...]] += data[feati]

            yield towrite

    def compute(self, loading):
        #TODO: Get rid of (or figure out) stride

        user = _compute(self.assn, loading, self.to2d,
                        self.n_frames, self.n_atoms,
                        self.solvent_ind, stride=1)

        return user

    def set_up_translation(self, deleted):
        """Turn indices from one form to another ('2d' -- '3d')

        :param deleted: Indices of states that were pruned
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

        self.to3d = to3d
        self.to2d = to2d
        return to3d, to2d

    def write_dat(self, data, features_to_select, out_fn_base, traj_fn=None,
                  top_fn=None, stride=1):

        dat_out_fn = "{}.dat".format(out_fn_base)
        tcl_out_fn = "{}.tcl".format(out_fn_base)

        howmany = self.n_frames // stride

        with open(dat_out_fn, 'w') as dat_f:
            for i, row in enumerate(
                    self.compute(data, features_to_select, stride)):
                [dat_f.write('{} '.format(d)) for d in row]
                dat_f.write('\n')

                if i % 10 == 0:
                    log.info("Done %d / %d", i, howmany)
                else:
                    log.debug("Done %d / %d", i, howmany)

        if traj_fn is not None and top_fn is not None:
            with open(tcl_out_fn, 'w') as tcl_f:
                tcl_f.write(VMDSCRIPT.format(top_fn=top_fn,
                                             traj_fn=traj_fn,
                                             step=stride,
                                             dat_fn=os.path.basename(
                                                 dat_out_fn)))


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

