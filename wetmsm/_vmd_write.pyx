
import numpy as np
cimport numpy as np


def  _compute(int[:, :] assn,
              float[:] loading,
              dict to2d,
              int n_frames,
              int n_atoms,
              int[:] solvent_ind,
              int stride):
    """Add "loading" to each relevant atom

    :param assn: (M,4) array 'assignments' file
        The columns are: frame, solvent, solute, shell (indices)

    :param loading: Values to apply to relevant features


    """
    cdef int fr, vent, ute_shell
    cdef float[:, :] user = np.zeros((n_frames, n_atoms), dtype=np.float)

    for i in range(assn.shape[0]):
        fr = assn[i, 0]
        vent = solvent_ind[assn[i, 1]]
        ute_shell = to2d[(assn[i, 2], assn[i, 3])]

        user[fr, vent] += loading[ute_shell]

    return user



