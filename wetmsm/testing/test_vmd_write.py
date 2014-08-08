__author__ = 'harrigan'

from unittest import TestCase
import unittest

import tables
from wetmsm.vmd_write import VMDWriter
import numpy as np
from numpy.testing import assert_array_equal


class TestVmdWrite(TestCase):
    def setUp(self):
        # Open file in memory and do not save
        assn_h = tables.open_file('tmp_assn.h5', 'w', driver='H5FD_CORE',
                                  driver_core_backing_store=0)
        assn = assn_h.create_earray(assn_h.root, 'assignments',
                                    atom=tables.UIntAtom(), shape=(0, 4))
        self.assn_h = assn_h


        # (frame, solvent, solute, shell)
        vent_a = 0
        vent_b = 1
        ute_1 = 0
        ute_2 = 1
        assn.append(np.array([
            [0, vent_a, ute_1, 0],
            [0, vent_b, ute_2, 0],
            [1, vent_a, ute_1, 1],
            [1, vent_b, ute_1, 1],
            [1, vent_a, ute_2, 1],
            [1, vent_b, ute_2, 1],
            [2, vent_a, ute_2, 0],
            [2, vent_b, ute_1, 0],
        ]))
        # O.   'O
        # O  :  O
        # O'   .O

        solvent_ind = np.array([2, 3])
        n_frames = 3
        n_atoms = 5
        n_solute = 2
        n_shells = 3

        self.vmd = VMDWriter(assn, solvent_ind, n_frames, n_atoms, n_solute,
                             n_shells)

    def test_add(self):
        loading2d = np.array([
            [2.0, 4.0, 99],
            [6.0, 8.0, 99]
        ])
        user = self.vmd.compute(loading2d)
        sb = np.zeros((3, 5))

        sb[:, 2:4] = np.array([
            [2, 6],
            [12, 12],
            [6, 2.0]
        ])

        assert_array_equal(user, sb)

    def test_max(self):
        loading2d = np.array([
            [2.0, 4.0, 99],
            [6.0, 8.0, 99]
        ])
        user = self.vmd.compute(loading2d, which='max')
        sb = np.zeros((3, 5))

        sb[:, 2:4] = np.array([
            [2, 6],
            [8, 8],
            [6, 2.0]
        ])

        assert_array_equal(user, sb)

    def test_avg(self):
        loading2d = np.array([
            [2.0, 4.0, 99],
            [6.0, 8.0, 99]
        ])
        user = self.vmd.compute(loading2d, which='avg')
        sb = np.zeros((3, 5))

        sb[:, 2:4] = np.array([
            [2, 6],
            [6, 6],
            [6, 2.0]
        ])

        assert_array_equal(user, sb)

    def tearDown(self):
        self.assn_h.close()


if __name__ == "__main__":
    unittest.main()
