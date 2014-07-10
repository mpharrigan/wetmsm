
import numpy as np
cimport numpy as np


def  _compute_chunk(
        unsigned int[:, :] assn,
        long[:] solvent_ind,
        dict to2d,
        double[:] loading,
        double[:, :] user):
    """Add "loading" to each relevant atom
    """
    cdef int fr, vent, ute_shell

    for i in range(assn.shape[0]):
        fr = assn[i, 0]
        vent = solvent_ind[assn[i, 1]]
        ute_shell = to2d[(assn[i, 2], assn[i, 3])]

        if ute_shell < 0:
            raise ValueError('Deleted shell')

        user[fr, vent] += loading[ute_shell]

