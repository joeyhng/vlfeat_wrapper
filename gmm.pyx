import numpy as np
cimport numpy as cnp
cnp.import_array()

ctypedef unsigned long long vl_uint64
ctypedef unsigned int vl_uint32
ctypedef vl_uint64 vl_size
ctypedef vl_uint32 vl_type

cdef extern from "vl/gmm.h":
    cdef struct VlGMM:
        pass
    VlGMM* vl_gmm_new (vl_type dataType, vl_size dimension, vl_size numComponents)
    double vl_gmm_cluster (VlGMM* self, void* data, vl_size numData)
    void* vl_gmm_get_means (VlGMM* self)
    void* vl_gmm_get_covariances (VlGMM* self)
    void* vl_gmm_get_priors (VlGMM* self)
    void* vl_gmm_get_posteriors (VlGMM* self)
    void  vl_gmm_set_means (VlGMM *self, void *means)
    void  vl_gmm_set_covariances (VlGMM *self, void *covariances)
    void  vl_gmm_set_priors (VlGMM *self, void *priors)
    void  vl_gmm_set_initialization (VlGMM *self, VlGMMInitialization init)


cdef extern from *:
    cdef struct VlKMeans:
        pass
    cdef enum VlGMMInitialization:
        VlGMMKMeans, VlGMMRand, VlGMMCustom 

    int VL_TYPE_DOUBLE


cdef class GMM:
    cdef int dimension
    cdef int n_clusters
    cdef VlGMM *thisptr

    def __cinit__(self, dimension, n_clusters, init='random'):
        self.dimension = dimension
        self.n_clusters = n_clusters
        self.thisptr = vl_gmm_new(VL_TYPE_DOUBLE, dimension, n_clusters)
        if init == 'kmeans':
            vl_gmm_set_initialization(self.thisptr, VlGMMKMeans)


    def __getstate__(self):
        return (self.dimension, self.n_clusters, 
                self.get_means(), self.get_covariances(), self.get_priors())

    def __setstate__(self, state):
        cdef cnp.ndarray[cnp.double_t, ndim=2, mode="c"] means, covariances
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] priors
        self.dimension, self.n_clusters, means, covariances, priors = state
        self.thisptr = vl_gmm_new(VL_TYPE_DOUBLE, self.dimension, self.n_clusters)
        vl_gmm_set_means(self.thisptr, <void*>means.data)
        vl_gmm_set_covariances(self.thisptr, <void*>covariances.data)
        vl_gmm_set_priors(self.thisptr, <void*>priors.data)

    def __reduce__(self):
        return GMM, (self.dimension, self.n_clusters), self.__getstate__()

    def cluster(self, cnp.ndarray[cnp.double_t, ndim=2, mode="c"] data):
        cdef int numData = data.shape[0]
        vl_gmm_cluster(self.thisptr, <void*>data.data, numData)
    
    def get_means(self):
        cdef double* means_ptr = <double*>vl_gmm_get_means(self.thisptr)
        cdef unsigned int i
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] means = np.zeros(self.n_clusters*self.dimension)
        for i in range(means.size):
            means[i] = means_ptr[i]
        return means.reshape((self.n_clusters, self.dimension))

    def get_covariances(self):
        cdef double* cov_ptr = <double*>vl_gmm_get_covariances(self.thisptr)
        cdef unsigned int i
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] cov = np.zeros(self.n_clusters*self.dimension)
        for i in range(cov.size):
            cov[i] = cov_ptr[i]
        return cov.reshape((self.n_clusters, self.dimension))

    def get_priors(self):
        cdef double* prior_ptr = <double*>vl_gmm_get_priors(self.thisptr)
        cdef unsigned int i
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] prior = np.zeros(self.n_clusters)
        for i in range(prior.size):
            prior[i] = prior_ptr[i]
        return prior

    def get_dimension(self):
        return self.dimension

    def get_n_clusters(self):
        return self.n_clusters


def test_gmm():
    cdef int numData = 1000
    cdef int dimension = 2
    cdef int n_clusters = 10
    cdef cnp.ndarray[cnp.double_t, ndim=2, mode="c"] data = np.random.randn(numData, dimension)
    for i in range(n_clusters):
        data[(i-1)*100:i*100, :] += np.random.rand(dimension) * 100
    gmm = GMM(dimension, n_clusters)
    gmm.cluster(data)
    m1, c1, p1 = gmm.get_means(), gmm.get_covariances(), gmm.get_priors()
    import pickle, os
    print 'saving:'
    pickle.dump(gmm, open('tmp.pkl', 'w'))
    print 'loading'
    gmm = pickle.load(open('tmp.pkl', 'r'))
    m2, c2, p2 = gmm.get_means(), gmm.get_covariances(), gmm.get_priors()
    print m1, m2
    print c1, c2
    print p1, p2
    print (m1-m2).sum(), (c1-c2).sum(), (p1-p2).sum()
    os.remove('tmp.pkl')

#test_gmm()
