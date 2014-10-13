"""Test solvent shell featurization

Author: Matthew Harrigan
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *
import unittest
import wetmsm
import numpy as np
import tempfile
import mdtraj as md
import tables
from os.path import join as pjoin
import mixtape.featurizer


def make_traj_from_1d(*particles):
    """Take numpy array and turn it into a one particle trajectory

    :param xyz: (n_frames, 3) np.ndarray
    """

    # Make dummy trajectory with enough atoms
    top = md.Topology()
    chain = top.add_chain()
    resi = top.add_residue(None, chain)
    for _ in particles:
        top.add_atom(None, None, resi)

    # Make xyz
    for_concat = []
    for p in particles:
        p = np.asarray(p)
        p3 = np.hstack((p[:, np.newaxis], np.zeros((len(p), 2))))
        p3 = p3[:, np.newaxis, :]
        for_concat += [p3]
    xyz = np.concatenate(for_concat, axis=1)
    traj = md.Trajectory(xyz, top)
    return traj


class TestMakeTraj(unittest.TestCase):
    def setUp(self):
        traj = make_traj_from_1d(
            [0, 0, 0, 0, 0, 5, 5, 5, 5],
            [-1, -2, -3, -4, -5, -6, -7, -8, -9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9]
        )
        self.traj = traj

    def test_make_traj(self):
        self.assertEqual(self.traj.n_frames, 9)
        self.assertEqual(self.traj.n_atoms, 3)


class TestShells(unittest.TestCase):
    def setUp(self):
        traj = make_traj_from_1d(
            [0, 0, 0, 0, 0, 5, 5, 5, 5],
            [-1, -2, -3, -4, -5, -6, -7, -8, -9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9]
        )
        self.tmpdir = tempfile.mkdtemp()
        self.traj_fn = pjoin(self.tmpdir, 'traj.h5')
        traj.save(self.traj_fn)

        shell_computation = wetmsm.SolventShellsComputation(
            solute_indices_fn=None,
            solvent_indices_fn=None, n_shells=3,
            shell_width=1, traj_fn=self.traj_fn, traj_top=None,
            counts_out_fn=pjoin(self.tmpdir, 'shell_count.h5'),
            assign_out_fn=pjoin(self.tmpdir, 'shell_assign.h5')
        )
        shell_computation.solute_indices = np.array([0])
        shell_computation.solvent_indices = np.array([1, 2])
        self.shell_comp = shell_computation

    def test_string_repr(self):
        self.assertEqual(str(self.shell_comp),
                         "Shells: {} with 3 shells of width 1 using None and None".format(
                             self.traj_fn))

    def test_featurization(self):
        self.shell_comp.featurize_all()

        count_f = tables.open_file(pjoin(self.tmpdir, 'shell_count.h5'))
        counts = count_f.root.shell_counts[:]

        should_be = np.array([
            [[2, 0, 0]],
            [[0, 2, 0]],
            [[0, 0, 2]],
            [[0, 0, 0]],
            [[0, 0, 0]],
            [[1, 0, 0]],
            [[0, 1, 0]],
            [[0, 0, 1]],
            [[0, 0, 0]]
        ])

        np.testing.assert_array_equal(counts, should_be)

        count_f.close()


class TestMixtape(unittest.TestCase):
    def setUp(self):
        traj = make_traj_from_1d(
            [0, 0, 0, 0, 0, 5, 5, 5, 5],
            [-1, -2, -3, -4, -5, -6, -7, -8, -9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9]
        )
        self.tmpdir = tempfile.mkdtemp()
        self.traj_fn = pjoin(self.tmpdir, 'traj.h5')
        traj.save(self.traj_fn)

        self.ssfeat = wetmsm.SolventShellsFeaturizer([0], [1, 2], 3, 1, True)


    def test_partial_transform(self):
        data, indices, fns = mixtape.featurizer.featurize_all([self.traj_fn],
                                                              self.ssfeat,
                                                              topology=None)

        norm = np.asarray([4 * np.pi * r ** 2 for r in [0.5, 1.5, 2.5]])
        should_be = np.array([
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ]) / norm

        np.testing.assert_array_equal(data, should_be)


if __name__ == "__main__":
    unittest.main()
