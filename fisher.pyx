import numpy as np
cimport numpy as np

ctypedef unsigned long long vl_uint64
ctypedef unsigned int vl_uint32
ctypedef vl_uint64 vl_size
ctypedef vl_uint32 vl_type


cdef extern from "vl/fisher.h":

    void vl_fisher_encode(void* enc, vl_type dataType, \
              void * means, vl_size dimension, vl_size numClusters, \
              void* covariances, void* priors, \
              void* data, vl_size numData, int flags)

    int VL_FISHER_FLAG_IMPROVED
    int VL_FISHER_FLAG_SQUARE_ROOT
    int VL_FISHER_FLAG_NORMALIZED
    int VL_FISHER_FLAG_FAST


def fisher_encode(gmm, np.ndarray[np.double_t, ndim=2, mode='c'] data, mode='improved'):
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] means = gmm.get_means() 
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] covariances = gmm.get_covariances()
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] priors = gmm.get_priors()
    cdef vl_size n_clusters = gmm.get_n_clusters()
    cdef vl_size dimension = gmm.get_dimension()
    cdef vl_size n_data = data.shape[0]
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] enc = np.zeros(2 * dimension * n_clusters)
    cdef int flag = 0
    if mode == 'fast':
        flag = VL_FISHER_FLAG_FAST
    elif mode == 'normalized':
        flag = VL_FISHER_FLAG_NORMALIZED
    elif mode == 'sqrt':
        flag = VL_FISHER_FLAG_SQUARE_ROOT
    elif mode == 'improved':
        flag = VL_FISHER_FLAG_IMPROVED
    vl_fisher_encode(<void*>enc.data, 2, 
            <void*>means.data, dimension, n_clusters, <void*>covariances.data, <void*>priors.data, 
            <void*>data.data, n_data, flag) 
    return enc


def signed_sqrt(np.ndarray[np.double_t, ndim=1, mode='c'] x):
    sign = np.sign(x)
    x = sign * np.sqrt(x * sign)
    return x


def normalize(np.ndarray[np.double_t, ndim=1, mode='c'] x):
    n = np.sqrt((x * x).sum())
    if n > 0:
        x /= n
    return x

def improve(np.ndarray[np.double_t, ndim=1, mode='c'] x):
    return normalize(signed_sqrt(x))

def test_fisher():
    cdef int numData = 10000
    cdef int dimension = 2
    cdef int numClusters = 10
    cdef np.ndarray[np.double_t, ndim=2, mode="c"] data = np.random.randn(numData, dimension)
    from gmm import GMM
    for i in range(numClusters):
        data[(i-1)*100:i*100, :] += np.random.rand(dimension) * 100
    gmm = GMM(dimension, numClusters)
    gmm.cluster(data)

    import pickle
    pickle.dump(gmm, open('tmp.pkl', 'w'))
    gmm = pickle.load(open('tmp.pkl', 'r'))

    '''
    import matplotlib.pyplot as plt
    plt.scatter( data[:,0], data[:, 1])
    plt.show()
    '''
    enc1 = fisher_encode(gmm, data[:1, :], mode='') 
    enc2 = fisher_encode(gmm, data[1:2, :], mode='') 
    enc3 = fisher_encode(gmm, data[:2, :], mode='') 

    print enc1+enc2
    print enc3
    print enc1.sum(), enc2.sum(), enc3.sum()
    print (((enc1+enc2)/2 - enc3)**2).sum()

    ienc3 = fisher_encode(gmm, data[:2, :]) 

    xenc3 = normalize(signed_sqrt((enc1 + enc2) / 2))
    print ienc3
    print xenc3
    print ((ienc3 - xenc3)**2).sum()


#test_fisher()
