
import numpy as np
cimport numpy as np
cimport cython

# Define helper-function type
ctypedef double (*hf_type)(double, double)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void  _compute_chunk(  
                            unsigned int[:, :] assn,
                            long[:] solvent_ind,
                            double[:, :] loading2d,
                            double[:, :] user,
                            int stride,
                            hf_type helper_func):
    """Apply a function to each line of assignments."""
    cdef int fr, vent, ute, shell

    for i in range(assn.shape[0]):
        # Get frame index with possible striding
        fr = assn[i, 0]
        if fr % stride == 0:
            fr /= stride
        else:
            continue
        
        # Apply helper function on our atom and shell of interest
        # to get new value for the atom
        vent = solvent_ind[assn[i, 1]]
        ute = assn[i, 2]
        shell = assn[i, 3]
        user[fr, vent] = helper_func(user[fr, vent], loading2d[ute, shell])


cdef double add_helper(double user, double loading):
    return user + loading


cdef double max_helper(double user, double loading):
    if loading > user:
        return loading
    else:
        return user

cdef double avg_counter(double user, double loading):
    return user + 1.0


@cython.boundscheck(False)
@cython.wraparound(False)
def  _compute_chunk_add(
        unsigned int[:, :] assn,
        long[:] solvent_ind,
        double[:, :] loading2d,
        double[:, :] user,
        int stride):
    """Add "loading" to each relevant atom"""
    _compute_chunk(assn, solvent_ind, loading2d, user, stride, add_helper)


@cython.boundscheck(False)
@cython.wraparound(False)
def  _compute_chunk_max(
        unsigned int[:, :] assn,
        long[:] solvent_ind,
        double[:, :] loading2d,
        double[:, :] user,
        int stride):
    """Compute max "loading" for each relevant atom"""
    _compute_chunk(assn, solvent_ind, loading2d, user, stride, max_helper)


@cython.boundscheck(False)
@cython.wraparound(False)
#@cython.cdivision(True)
def  _compute_chunk_avg(
        unsigned int[:, :] assn,
        long[:] solvent_ind,
        double[:, :] loading2d,
        double[:, :] user,
        int stride,
        double[:, :] occ):
    """Add "loading" and average by number of shells it's in.
    
    :param occ: Running occupancy count
    """
    _compute_chunk(assn, solvent_ind, loading2d, user, stride, add_helper)
    _compute_chunk(assn, solvent_ind, loading2d, occ,  stride, avg_counter)

    cdef int i, j
    cdef double uu
    for i in range(user.shape[0]):
        for j in range(user.shape[1]):
            uu = user[i, j]
            if uu > 1e-5:
                user[i, j] /= occ[i, j]


