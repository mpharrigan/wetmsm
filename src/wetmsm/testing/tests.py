__author__ = 'harrigan'

import unittest
import numpy as np
import wetmsm


class TestAnalysis(unittest.TestCase):
    def setUp(self):
        self.fp3d = np.array([
            [[0, 1, 0, 0]],
            [[0, 0, 2, 0]],
            [[0, 0, 0, 4]]
        ])

        self.fp2d = np.array([
            [0, 1, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 4]
        ])

    def test_reshape(self):
        fp2d = wetmsm.analysis.reshape(self.fp3d)
        np.testing.assert_array_equal(fp2d, self.fp2d)

    def test_prune(self):
        should_be = np.array([
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 4]
        ])

        actually_is = wetmsm.analysis.prune(self.fp2d)
        np.testing.assert_array_equal(should_be, actually_is)

    def test_normalize(self):
        should_be = np.array([
            [[0, 1 / (1.5 ** 2), 0, 0]],
            [[0, 0, 2 / (2.5 ** 2), 0]],
            [[0, 0, 0, 4 / (3.5 ** 2)]]
        ]) / ( 4 * np.pi)

        actually_is = wetmsm.analysis.normalize(self.fp3d, shell_w=1)
        np.testing.assert_array_almost_equal(should_be, actually_is)


class TestAnalysis2(unittest.TestCase):
    def setUp(self):
        self.fp3d = np.array([
            [[0, 1, 0, 0], [44, 43, 42, 41]],
            [[0, 0, 1, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 1], [0, 0, 0, 0]]
        ])

        self.fp2d = np.array([
            [0, 1, 0, 0, 44, 43, 42, 41],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])

    def test_reshape(self):
        fp2d = wetmsm.analysis.reshape(self.fp3d)
        np.testing.assert_array_equal(fp2d, self.fp2d)

    def test_prune(self):
        should_be = np.array([
            [1, 0, 0, 44, 43, 42, 41],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0]
        ])

        actually_is = wetmsm.analysis.prune(self.fp2d)
        np.testing.assert_array_equal(should_be, actually_is)


class TestPruneAll(unittest.TestCase):
    def setUp(self):
        self.fp2d1 = np.array([
            [0, 1, 0, 0, 9],
            [0, 0, 2, 0, 9],
            [0, 0, 0, 4, 9]
        ])

    def test_delete(self):
        fp2d2 = np.copy(self.fp2d1)
        fp2d2[:, -1] = 8

        out1, out2 = wetmsm.analysis.prune_all([self.fp2d1, fp2d2])

        self.assertEqual(out1.shape[1], 3)
        self.assertEqual(out2.shape[1], 3)
        self.assertEqual(out1.shape[0], 3)
        self.assertEqual(out2.shape[0], 3)

    def test_dontdelete(self):
        fp2d2 = np.copy(self.fp2d1)
        fp2d2[1, 0] = 83

        out1, out2 = wetmsm.analysis.prune_all([self.fp2d1, fp2d2])

        self.assertEqual(out1.shape[1], 4)
        self.assertEqual(out2.shape[1], 4)
        self.assertEqual(out1.shape[0], 3)
        self.assertEqual(out2.shape[0], 3)
