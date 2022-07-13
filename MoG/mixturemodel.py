import numpy as np


class mixtureModel:

    def __init__(self, n_components=2):
        self.k = n_components

    def fit(self, X):

        self.X = X

        self.n_samples, self.n_features = self.X.shape

        # Binary indicator (n_s , K) if sample i is in k comp then self.z[i,k] = 1
        self.z = np.zeros((self.n_samples, self.k), dtype=np.float32)
        self.idx = np.random.choice(self.k, size=self.n_samples)
        for i, idx in enumerate(self.idx):
            self.z[i, idx] = 1

        # Weight set
        self.a = np.sum(self.z, axis=0) / self.n_samples
        # Initialize the mean vector for each component
        self.mu = np.random.uniform(size=(self.k, self.n_features))

        # Initialize a covariance matrix for each component
        self.cov = np.stack([np.cov(X.T) for _ in range(self.k)])

        # E-step
        self._e_step()
        self._m_step()

    def _e_step(self):
        self.mem_weights = np.asarray([[self._get_membership_weight(i, j) for j in range(
            self.k)] for i in range(self.n_samples)], dtype=np.float64)
        # print(self.mem_weights.shape)

    def _m_step(self):

        # Update a
        N_k = np.sum(self.mem_weights, axis=0)
        self.a = N_k / self.n_samples

        # Update means and cov Matrices
        # TODO : Turn this into vectorized code .... if i have time ;)
        for k in range(self.k):
            # Initialize temp arrays
            mu_temp = np.zeros(self.n_features)
            cov_temp = np.zeros((self.n_features, self.n_features), dtype=np.float64)
            # Loop over all samples
            for i in range(self.n_samples):
                mu_temp += self.mem_weights[i, k] @ self.X[i]
                cov_temp += (self.X[i] - self.mu[k]) @ (self.X[i] - self.mu[k]).T  * self.mem_weights[i,k]

            # Complete the M step
            mu_temp /= N_k
            cov_temp /= N_k[k]
            self.mu[k] = mu_temp
            self.cov[k] = cov_temp


    def _get_multivariate_gaussian_density(self, x, mean_vector, cov_matrix, z_k=1.0):

        den = 1 / (pow((2 * np.pi), len(mean_vector) / 2)
                   * np.sqrt(np.linalg.det(cov_matrix)))
        exp_exp = (-1 / 2) * (x -
                              mean_vector).T @ np.linalg.inv(cov_matrix) @ (x - mean_vector)

        return np.exp(exp_exp) * den

    def _get_membership_weight(self, i, k):
        num = self._get_multivariate_gaussian_density(
            self.X[i], self.mu[k], self.cov[k], z_k=self.z[i, k]) * self.a[k]
        den = .0
        for m in range(self.k):
            den += self._get_multivariate_gaussian_density(
                self.X[i], self.mu[m], self.cov[m], z_k=self.z[i, m]) * self.a[m]

        return num / den

    def _get_log_likelihood(self):
        pass
