import numpy as np
cimport numpy as cnp
cnp.import_array()

ctypedef unsigned long long vl_uint64
ctypedef unsigned int vl_uint32
ctypedef vl_uint64 vl_size
ctypedef vl_uint32 vl_type

cdef extern from *:
    cdef enum VlVectorComparisonType:
        VlDistanceL1,       
        VlDistanceL2,    
        VlDistanceChi2, 
        VlDistanceHellinger,
        VlDistanceJS,    
        VlDistanceMahalanobis,
        VlKernelL1,       
        VlKernelL2,     
        VlKernelChi2,
        VlKernelHellinger,   
        VlKernelJS        

cdef extern from "vl/kmeans.h":

    cdef struct VlKMeans:
        pass

    cdef enum VlKMeansAlgorithm:
        VlKMeansLloyd,
        VlKMeansElkan,
        VlKMeansANN

    VlKMeans* vl_kmeans_new (vl_type dataType, VlVectorComparisonType distance)
    double  vl_kmeans_cluster (VlKMeans *self, void *data, vl_size dimension, vl_size numData, vl_size numCenters)
    void    vl_kmeans_quantize (VlKMeans *self, vl_uint32 *assignments, void *distances, void *data, vl_size numData)
    void    vl_kmeans_set_centers (VlKMeans* self, void *centers, vl_size dimension, vl_size numCenters)
    void *  vl_kmeans_get_centers  (VlKMeans * self)   
    void    vl_kmeans_set_algorithm (VlKMeans *self, VlKMeansAlgorithm algorithm)



# TODO: destructor?
# may have memory leaks now

cdef class KMeans:
    cdef int dimension
    cdef int n_clusters
    cdef VlKMeans *thisptr

    def __cinit__(self, dimension, n_clusters):
        self.dimension = dimension
        self.n_clusters = n_clusters
        self.thisptr = vl_kmeans_new(2, VlDistanceL2)
        vl_kmeans_set_algorithm(self.thisptr, VlKMeansANN)

    def __getstate__(self):
        cdef cnp.ndarray[cnp.double_t, ndim=2, mode="c"] centers = \
                cnp.PyArray_SimpleNewFromData(2, [self.n_clusters, self.dimension], cnp.NPY_DOUBLE, 
                                                vl_kmeans_get_centers(self.thisptr))
        return (self.dimension, self.n_clusters, centers)

    def __setstate__(self, state):
        cdef cnp.ndarray[cnp.double_t, ndim=2, mode="c"] centers 
        self.dimension, self.n_clusters, centers = state
        self.thisptr = vl_kmeans_new(2, VlDistanceL2)
        vl_kmeans_set_centers(self.thisptr, <void*>centers.data, self.dimension, self.n_clusters)

    def __reduce__(self):
        return KMeans, (self.dimension, self.n_clusters), self.__getstate__()

    def cluster(self, cnp.ndarray[cnp.double_t, ndim=2, mode="c"] data):
        cdef int n_data = data.shape[0]
        vl_kmeans_cluster(self.thisptr, <void*>data.data, self.dimension, n_data, self.n_clusters)

    def quantize(self, cnp.ndarray[cnp.double_t, ndim=2, mode="c"] data):
        cdef int n_data = data.shape[0]
        cdef cnp.ndarray[vl_uint32, ndim=1, mode='c'] result = np.zeros(n_data, dtype=np.uint32)
        vl_kmeans_quantize(self.thisptr, <vl_uint32*>result.data, NULL, <void*>data.data, n_data)
        return result

    def predict(self, cnp.ndarray[cnp.double_t, ndim=2, mode="c"] data):
        return self.quantize(data)

    def get_means(self):
        cdef double* means_ptr = <double*>vl_kmeans_get_centers(self.thisptr)
        cdef unsigned int i
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] means = np.zeros(self.n_clusters * self.dimension)
        for i in range(means.size):
            means[i] = means_ptr[i]
        return means.reshape((self.n_clusters, self.dimension))

    def get_n_clusters(self):
        return self.n_clusters

    def get_dimension(self):
        return self.dimension


# code for constructing numpy array from c pointer
# method 1:
#cdef cnp.ndarray[cnp.double_t, ndim=2, mode="c"] centers2 = \
#cnp.PyArray_SimpleNewFromData(2, [self.n_clusters, self.dimension], cnp.NPY_DOUBLE, vl_kmeans_get_centers(self.thisptr))
# method 2:
#cdef cnp.ndarray[cnp.double_t, ndim=2, mode="c"] centers = np.zeros((self.n_clusters, self.dimension))
#centers.data = <char*>vl_kmeans_get_centers(self.thisptr)

def test_kmeans():
    cdef int numData = 1000
    cdef int dimension = 2 
    cdef int numClusters = 10
    cdef cnp.ndarray[cnp.double_t, ndim=2, mode="c"] data = np.random.randn(numData, dimension)
    for i in range(numClusters):
        data[(i-1)*100:i*100, :] += np.random.rand(dimension) * 50
    kmeans = KMeans(dimension, numClusters)
    kmeans.cluster(data)
    result = kmeans.quantize(data)
    print result
    import pickle
    pickle.dump(kmeans, open('tmp.pkl', 'w'))
    kmeans2 = pickle.load(open('tmp.pkl', 'r'))
    print kmeans2.quantize(data)

    # visualize the result
    rand_color = np.random.rand(numClusters, 3)
    import matplotlib.pyplot as plt
    for i in range(numClusters):
        idx = result == i
        plt.scatter(data[idx,0], data[idx,1], color=rand_color[i, :])
    plt.show()


#test_kmeans()
