
import numpy as np
cimport numpy as np


def  _compute_chunk(unsigned int[:, :] assn,
              long[:] solvent_ind,
              dict to2d,
              double[:] loading,
              double[:, :] user):
    """Add "loading" to each relevant atom

    :param assn: (M,4) array 'assignments' file
        The columns are: frame, solvent, solute, shell (indices)

    :param loading: Values to apply to relevant features


    """
    cdef int fr, vent, ute_shell

    for i in range(assn.shape[0]):
        fr = assn[i, 0]
        vent = solvent_ind[assn[i, 1]]
        ute_shell = to2d[(assn[i, 2], assn[i, 3])]

        user[fr, vent] += loading[ute_shell]




