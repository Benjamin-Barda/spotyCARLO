import numpy as np
# from node import Node
from DecisionTree.node import Node



class Tree : 
    """
    This class provides the methods to create, tune, and use a simple decision tree 
    classification Algorithm
    """

    def __init__(self, max_depth = 6, min_sample_per_split=2, inForest = False, random_seed = None) : 
        """
        Args: 
            {int} max_depth : Max depth the tree can reach 
            {int} min_sample_per_split : Minimum number of samples in order to be able to perform a split in a node
        Return: 
            {Tree} self : Initialize the Tree object
        """

        self.in_forest = inForest

        self.max_depth = max_depth 
        self.min_sample_per_split = min_sample_per_split
        self.root = None

    
        if random_seed is not None : 
            np.random.seed(random_seed)

    def _split(self, X, thresh) : 
        """
        Split on X based on thresh

        Args : 
            {np.ndarray} X : Observations 
            {int}   thresh : value of split
        Return : 
            {np.array} left_idx  : where X <= thresh
            {np.array} right_idx : where X > thresh
        """

        left_idx = np.argwhere(X <= thresh).flatten()
        right_idx = np.argwhere(X > thresh).flatten()

        return (left_idx, right_idx)   

    def _entropyImpurity(self, Y) : 
        """
        Calculate the entropy impurity on Y

        Args: 
            {np.array} Y : Lables on which to calculate entropy impurity
        Return: 
            {float32} entropy : Entropy impurity (ref. https://en.wikipedia.org/wiki/Entropy_(information_theory) )

        """
        prop =  np.bincount(Y) / Y.shape[0]
        entropy = -np.sum([a * np.log2(a) for a in prop if a > 0])
        return entropy

    def _infGain(self, X, y, thresh) : 
        """
        Calculate the information gain.

        Args: 
            {np.ndarray} X : Observations
            {np.array}   y : Labels for X's observations
            {int}   thresh : Threshhold on which to calculate the split
        Return: 
            {int} information Gain : Information gain after the split on thresh (ref https://en.wikipedia.org/wiki/Information_gain_in_decision_trees )
        """

        my_loss = self._entropyImpurity(y)


        l_idx, r_idx = self._split(X, thresh)

        n, n_l, n_r = len(y), len(l_idx), len(r_idx)

        if n_l == 0 or n_r == 0: 
            return 0

        chid_loss = (n_l / n) * self._entropyImpurity(y[l_idx]) + (n_r / n) * self._entropyImpurity(y[r_idx])

        return my_loss - chid_loss
    
    def _bestSplit(self, X, y, features) :
        """
        Args: 
            {np.ndarray}      X : Observations
            {np.array}        y : Labels for X's observations
            {np.array} features : Features to be considered on which to look for the best split
        Return:
            {tuple(int, int)} : Feature and Threshold that best split our data based on IG.
        """
        split = {
            'score' : -1, 
            'feature' : None, 
            'threshold' : None
        }

        for feature in features : 
            X_feat = X[:, feature]
            thresholds = np.unique(X_feat)
            #TODO : try split not 
            # T <- linspace 
            for t in thresholds : 
                score = self._infGain(X_feat, y, t)
                if score > split['score'] : 
                    split['score'] = score
                    split['feature'] = feature
                    split['threshold'] = t

        return split['feature'], split['threshold']


    # Helper function to stop recursion 
    def _finished(self, depth) : 
        if depth >= self.max_depth or self.n_samples < self.min_sample_per_split or self.n_classes == 1 : 
            return True
        return False

    def _build(self, X, y, depth = 0) : 
        """
        Recursive function that build the Tree

        Args: 
            {np.ndarray} X : Observations
            {np.array}   y : Labels for X's observations
            {int}    depth : Current depth of the three
        Return: 
            {Tree} root : Save in self.root the root of the tree 
        """
        self.n_samples, self.n_features = X.shape
        self.n_classes = len(np.unique(y))


        # base case
        if self._finished(depth) :
            # This try except is more of a temp-fix ... for now it does the job but we need to check better
            try:
                return Node(value = np.argmax(np.bincount(y)))
            except(ValueError): 
                print("FAILED - EMPTY SEQ HANDLED")
                return Node()
        

        if self.in_forest:
            feats = np.random.choice(self.n_features, round(np.sqrt(self.n_features)), replace=False) 
        else: 
            feats = np.random.choice(self.n_features, self.n_features, replace=False)

        best_feat, best_thresh = self._bestSplit(X, y, feats)

        #recursive step
        l_idx, r_idx = self._split( X[:, best_feat], best_thresh)

        l = self._build( X[l_idx, :], y[l_idx], depth + 1)
        r = self._build(X[r_idx, :], y[r_idx], depth + 1)

        return Node(best_feat, best_thresh, l, r)
    
    def fit(self, X, y): 
        """
        Build the tree

        Args: 
            {np.ndarray} X : Observations
            {np.array}   y : Labels for X's observations
        Return: 
            {Tree} root : Save in self.root the root of the tree 
        """
        self.root = self._build(X,y)
    

    def _traverse(self, x, node): 
        """
        Recursive function to explore the tree

        Args: 
            {np.ndarray} x : Observation
            {np.array}   y : Labels for X's observations
        Return: 
            {int} prediction for x
        """

        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.thresh:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X, trgt = None) :
        """
        Bulk prediction 

        Args: 
            {np.ndarray} X : Observations
        Return: 
            {list} predictions : list of predictions on X
        """ 
        predictions = [self._traverse(x, self.root) for x in X]

        if trgt != None : 
            assert trgt.shape[0] == len(predictions) 
            acc = sum(predictions == trgt) / trgt.shape[0]

            return predictions, acc

        return predictions
        
    def __str__(self) :

        s = "tree" 
        return (s)


        
        