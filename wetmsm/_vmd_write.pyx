
import numpy as np
cimport numpy as np


def  _compute(int[:, :] assn,
              float[:] loading,
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
        ute_shell = assn[i, 2]

        user[fr, vent] += loading[ute_shell]

    return user


def  _translate(int[:, :] assn,
                         dict to2d):

    cdef int[:, :] assn_out = assn[:, 0:3]
    cdef int ute_shell

    for i in range(assn.shape[0]):
        ute_shell = to2d[(assn[i, 2], assn[i, 3])]
        assn_out[i, 2] = ute_shell

    return assn_out



