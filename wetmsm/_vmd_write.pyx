
import numpy as np
cimport numpy as np


def  _compute_chunk(
        unsigned int[:, :] assn,
        long[:] solvent_ind,
        double[:] loading2d,
        double[:, :] user):
    """Add "loading" to each relevant atom
    """
    cdef int fr, vent, ute, shell

    for i in range(assn.shape[0]):
        fr = assn[i, 0]
        vent = solvent_ind[assn[i, 1]]
        ute = assn[i, 2]
        shell = assn[i, 3]

        user[fr, vent] += loading2d[ute, shell]

