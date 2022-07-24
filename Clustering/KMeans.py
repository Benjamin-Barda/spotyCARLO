import numpy as np
from numpy import linalg as la

class KMeans_plus_plus:
    """
    1. furthest first when string is "in"  "furthest_first"
    2. kmeans         when string is either None or equals kmeans
    2. kmeans++       For any other string
    """
    
    def __init__(self, K=4, init="kmeans++", rand_seed=None):
        self.K = K
        self.init = init
        self.rand_seed = rand_seed
    
        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # the centers (mean feature vector) for each cluster
        self.centroids = []
        self.__set_seed()

    ##########################################################################################
    # kmeans furthest first initialization
    def __initialize_centroids_further_first(self):
        
        centroids = list( self.__random_centroid(how_many_centroids=1) )
    
        ## compute the remaining k - 1 centroids
        for _ in range(self.K - 1):
            next_cent_id = self.__max_arg(centroids)
            centroids.append(self.X[next_cent_id, :])
        return np.array(centroids)

    #kmeans++ initialization
    def __plus_plus_init(self):
        # (K, d) 
        centroids = np.zeros((self.K, self.X.shape[1]), dtype=np.float32)
        # First centroid | note randint sample from discrete uniform
        centroids[0,:] = self.X[np.random.randint(low = 0, high = self.X.shape[0], size = 1), : ]

        for k in range (1, self.K) : 
            # Calculate the distance to the closest centroid for each point
            d = np.asarray([self.__min_dist(point, centroids[:k,:]) for point in self.X])
            d_sum = np.sum(d)

            # Turn the distance into probabilities that sums up to one
            d /= d_sum
        
            # sample with distance as weight the new centroid
            centroids[k, :] = self.X[np.random.choice(self.X.shape[0], p=d),:]

        return centroids         
    
    def __max_arg(self, centroids):
        return np.argmax([ self.__min_dist(point,centroids ) for point in self.X])
        
    def __min_dist(self, point, centroids):
        return min( [ self.euclidean_distance(point, centroid)**2 for centroid in centroids] ) 
    
    ##########################################################################################
    # get the how_many_centroids "centroids". 1 if k-means++, but k if k-means
    def __random_centroid(self, how_many_centroids):
        #self.__set_seed() 
        l = self.X.shape[0]
        choices = np.random.choice(l, how_many_centroids)
        return np.array(self.X[choices, :])
    
    # set the random seed for reproducibility
    def __set_seed(self):
        if self.rand_seed is not None and isinstance(self.rand_seed, int):
            np.random.seed(self.rand_seed)
    ##########################################################################################
    def fit_predict(self, X):
        self.X = X
        # print(self.X.shape)
        self.n_samples, self.n_features = X.shape

        #Initialisations...
        if self.init in 'further_first': 
            self.centroids = self.__initialize_centroids_further_first()
        elif self.init is None or self.init == "kmeans": 
            self.centroids = self.__random_centroid(how_many_centroids=self.K)
        # Kmean ++
        else: 
            self.centroids = self.__plus_plus_init()

        
                    
        #Initialize an empty array to compare with current centroids to see if convergence occurs...
        i, j = self.centroids.shape
        centroids_old = np.zeros(shape=(i, j) )

        #if the current centroids are the same as the previous centroids, then stop
        while not self.__has_converged(centroids_old, self.centroids) :
            
            # Assign samples to closest centroids (create clusters)
            self.clusters = self.__create_clusters(self.centroids)

            # Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self.__centroids_means_of_clusters(self.clusters)

        # Classify samples as the index of their clusters
        return self.__get_cluster_labels(self.clusters)

    # each sample will get the label of the cluster it was assigned to
    def __get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            labels[cluster] = cluster_idx
        return labels
    
    # Assign the samples to the closest centroids to create clusters; Expectation step
    def __create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self.__closest_centroid_label(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters
    
    # arg that minimizes the distance of the current sample to each centroid
    def __closest_centroid_label(self, point, centroids):
        return np.argmin( [ self.euclidean_distance(point, centroid)**2 for centroid in centroids] ) 

    # assign mean value of clusters to centroids; Maximisation step
    def __centroids_means_of_clusters(self, clusters):
        return [ np.mean(self.X[cluster], axis=0) for cluster in clusters]
        
    # distances between each old and new centroids
    def __has_converged(self, old_centroids, current_centroids):
        distances = [self.euclidean_distance(old_centroids[i], current_centroids[i])**2 for i in range(self.K)]
        return sum(distances) == 0
    
    def euclidean_distance(self, point1, point2):
        dist = la.norm(point1 - point2)
        return dist
