
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def  _compute_chunk_add(
        unsigned int[:, :] assn,
        long[:] solvent_ind,
        double[:, :] loading2d,
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


@cython.boundscheck(False)
@cython.wraparound(False)
def  _compute_chunk_max(
        unsigned int[:, :] assn,
        long[:] solvent_ind,
        double[:, :] loading2d,
        double[:, :] user):
    """Compute max "loading" for each relevant atom
    """
    cdef int fr, vent, ute, shell

    for i in range(assn.shape[0]):
        fr = assn[i, 0]
        vent = solvent_ind[assn[i, 1]]
        ute = assn[i, 2]
        shell = assn[i, 3]

        if loading2d[ute, shell]  > user[fr, vent]:
            user[fr, vent] = loading2d[ute, shell]


@cython.boundscheck(False)
@cython.wraparound(False)
def  _compute_chunk_avg(
        unsigned int[:, :] assn,
        long[:] solvent_ind,
        double[:, :] loading2d,
        double[:, :] user):
    """Add "loading" and average by number of shells it's in.
    """
    cdef int fr, vent, ute, shell
    cdef double[:, :] counts = np.zeros_like(user)

    for i in range(assn.shape[0]):
        fr = assn[i, 0]
        vent = solvent_ind[assn[i, 1]]
        ute = assn[i, 2]
        shell = assn[i, 3]

        user[fr, vent] += loading2d[ute, shell]
        counts[fr, vent] += 1.0

    for i in range(user.shape[0]):
        for j in range(user.shape[1]):
            if counts[i, j] > 0:
                user[i, j] /= counts[i, j]

