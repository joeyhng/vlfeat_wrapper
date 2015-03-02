import numpy as np
cimport numpy as np

ctypedef unsigned long long vl_uint64
ctypedef unsigned int vl_uint32
ctypedef vl_uint64 vl_size
ctypedef vl_uint32 vl_type


cdef extern from "vl/vlad.h":

    void vl_vlad_encode (void* enc, vl_type dataType,
                         void* means, vl_size dimension, vl_size numClusters,
                         void* data, vl_size numData, void* assignments,
                         int flags)       

    int VL_VLAD_FLAG_NORMALIZE_COMPONENTS
    int VL_VLAD_FLAG_SQUARE_ROOT
    int VL_VLAD_FLAG_UNNORMALIZED
    int VL_VLAD_FLAG_NORMALIZE_MASS


def vlad_encode(kmeans, np.ndarray[np.double_t, ndim=2, mode='c'] data, mode=''):
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] means = kmeans.get_means() 
    cdef vl_size n_clusters = kmeans.get_n_clusters()
    cdef vl_size dimension = kmeans.get_dimension()
    cdef vl_size n_data = data.shape[0]
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] enc = np.zeros(dimension * n_clusters)
    cdef unsigned int i
    cdef int flag = 0

    if 'components' in mode:
        flag |= VL_VLAD_FLAG_NORMALIZE_COMPONENTS
    if 'sqrt' in mode:
        flag |= VL_VLAD_FLAG_SQUARE_ROOT
    if 'mass' in mode:
        flag |= VL_VLAD_FLAG_NORMALIZE_MASS
    if 'unnormalized' in mode:
        flag |= VL_VLAD_FLAG_UNNORMALIZED

    cdef np.ndarray[np.uint32_t, ndim=1, mode='c'] idx = kmeans.quantize(data)
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] assignments = np.zeros(data.shape[0] * n_clusters)
    for i in range(n_data):
        assignments[i * n_clusters + idx[i]] = 1

    vl_vlad_encode(<void*>enc.data, 2,
                   <void*>means.data, dimension, n_clusters,
                   <void*>data.data, n_data, <void*>assignments.data, flag)
    return enc


