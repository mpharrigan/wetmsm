__author__ = 'harrigan'

import numpy as np
import logging

log = logging.getLogger()

EPS = 1e-7


def reshape(fp3d):
    """Reduce 3d array to 2d.

    We start with indices (frame, solute, shell) and convert to
    (frame, {solute*shell}), or alternatively (frame, feature)
    """
    n_frame, n_solute, n_shell = fp3d.shape
    fp2d = np.reshape(fp3d, (n_frame, n_solute * n_shell))
    return fp2d


def prune(fp2d):
    """Get rid of features that are always zero."""

    to_delete = np.where(np.all(fp2d == 0, axis=0))
    log.info('Trimming %d features', len(to_delete[0]))
    fp2d_pruned = np.delete(fp2d, to_delete, axis=1)
    return fp2d_pruned


def prune_all(fp2d_all):
    """Prune a list of feature trajectories.

    Only remove a feature if it is zero in *all* trajectories.
    """

    assert len(fp2d_all) > 0, 'We expect a list'
    n_features = fp2d_all[0].shape[1]

    zero_variance = np.zeros((len(fp2d_all), n_features), dtype=bool)

    for i, fp2d in enumerate(fp2d_all):
        assert fp2d.shape[1] == n_features, 'Constant num features.'
        zero_variance[i, :] = np.var(fp2d, axis=0) < EPS

    to_delete = np.where(np.all(zero_variance, axis=0))
    log.info('Trimming %d features from all trajectories', len(to_delete[0]))
    fp2d_all_pruned = [np.delete(fp2d, to_delete, axis=1)
                       for fp2d in fp2d_all]
    return fp2d_all_pruned


def normalize(fp3d, shell_w):
    """Normalize by 4 pi r^2 dr.

    :param shell_w: Shell width
    """
    _, _, n_shell = fp3d.shape
    shell_edges = np.linspace(0, shell_w * (n_shell), num=(n_shell + 1))
    shell_mids = (np.diff(shell_edges) / 2) + shell_edges[:-1]
    norm = 4 * np.pi * (shell_mids ** 2) * shell_w
    norm = norm[np.newaxis, np.newaxis, :]
    fp3d_norm = fp3d / norm
    return fp3d_norm


