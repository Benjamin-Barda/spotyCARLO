import numpy as np 



class mixtureModel: 

    def __init__(self, n_components = 2) : 
        self.k = n_components


    
    def fit(self, X, y): 
        n_samples, n_features = X.shape


        # Binary indicator (n_s , K) if sample i is in k comp then self.z[i,k] = 1
        self.z = np.zeros((n_samples, self.k))
        self.idx = np.random.choice(self.k, size = n_samples)
        for i, idx in enumerate(self.idx) : 
            self.z[i,idx] = 1
        
        # Weight set
        self.a = np.sum(self.z, axis = 0) / n_samples

        # Initialize the mean vector for each component
        self.mu = np.random.uniform(size=(self.k, n_features))

        # Initialize a covariance matrix for each component
        self.cov = np.stack([np.cov(X.T) for _ in range(self.k)])


        c = self.get_membership_weight(X[3], 1, 1)      



    def _get_multivariate_gaussian_density(self, x, mean_vector, cov_matrix, z_k = 1) : 

        if z_k == 0 :
            return 0

        den = 1 / (pow((2 * np.pi), len(mean_vector) / 2) * np.sqrt(np.linalg.det(cov_matrix)))
        exp_exp = (-1 / 2) * (x - mean_vector).T @ np.linalg.inv(cov_matrix) @ (x - mean_vector)

        return np.exp(exp_exp) * den
    


    def get_membership_weight(self, x_i, i, k ) : 
        num = self._get_multivariate_gaussian_density(x_i, self.mu[k], self.cov[k], z_k = self.z[i,k]) * self.a[k]
        den = .0
        for m in range(self.k) : 
            den += self._get_multivariate_gaussian_density(x_i, self.mu[m], self.cov[m], z_k = self.z[i,m]) * self.a[m]

        return num / den


        

        




