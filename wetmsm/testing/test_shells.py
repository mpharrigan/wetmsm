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
import os
from os.path import join as pjoin
import subprocess
from msmbuilder.dataset import dataset


def make_traj_from_1d(*particles):
    """Take numpy arrays and turn it into trajectories in x / particle

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

        shell_computation = wetmsm.SolventShellsFeaturizer(
            n_shells=3, shell_width=1, solute_indices=np.array([0]),
            solvent_indices=np.array([1, 2])
        )
        self.shell_comp = shell_computation


    def test_featurization(self):
        counts = self.shell_comp.partial_transform(md.load(self.traj_fn))

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

        np.testing.assert_array_equal(counts, should_be)


class TestMSMBuilder(unittest.TestCase):
    def setUp(self):
        traj = make_traj_from_1d(
            [0, 0, 0, 0, 0, 5, 5, 5, 5],
            [-1, -2, -3, -4, -5, -6, -7, -8, -9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9]
        )
        self.tmpdir = tempfile.mkdtemp()
        self.traj_fn = pjoin(self.tmpdir, 'traj.h5')
        self.outfn = pjoin(self.tmpdir, 'feat')
        traj.save(self.traj_fn)

        self.ute_fn = pjoin(self.tmpdir, 'ute')
        self.vent_fn = pjoin(self.tmpdir, 'vent')
        np.savetxt(self.ute_fn, np.array([0]), fmt="%d")
        np.savetxt(self.vent_fn, np.array([1, 2]), fmt="%d")


    def test_partial_transform(self):
        with open(os.devnull) as dn:
            subprocess.call(
                [
                    'msmb', 'SolventShellsFeaturizer', '--trjs', self.traj_fn,
                    '--solute_indices', self.ute_fn, '--solvent_indices',
                    self.vent_fn, '--n_shells', '3', '--shell_width', '1',
                    '--out', self.outfn
                ], stdout=dn, stderr=dn
            )
        data = dataset(self.outfn)[0]

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

    def test_assign(self):
        with open(os.devnull) as dn:
            subprocess.call(
                [
                    'msmb', 'SolventShellsAssigner', '--trjs', self.traj_fn,
                    '--solute_indices', self.ute_fn, '--solvent_indices',
                    self.vent_fn, '--n_shells', '3', '--shell_width', '1',
                    '--out', self.outfn, '--chunk', '2'
                ], stdout=dn, stderr=dn
            )

        data = dataset(self.outfn)[0]

        should_be = np.array([
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 1],
            [1, 1, 0, 1],
            [2, 0, 0, 2],
            [2, 1, 0, 2],
            # 3
            # 4
            [5, 1, 0, 0],
            [6, 1, 0, 1],
            [7, 1, 0, 2],
            # 8
        ])

        np.testing.assert_array_equal(data, should_be)


if __name__ == "__main__":
    unittest.main()
