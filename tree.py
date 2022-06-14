import numpy as np 
import pandas as pd

dF = pd.read_csv('data//csvs//dataframeV1.csv', index_col=0)
dF = dF.drop(['id', 'uri'], axis = 1)
dF.head()

dF.label = pd.Categorical(dF.label)
dF['Y'] = dF.label.cat.codes
dF = dF.drop(['label'], axis = 1)
dF.head()


class Node : 

    def __init__(self, feature = None, thresh = None,  left = None, right = None, value = None) : 

        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self.thresh = thresh
    
    def is_leaf(self): 
        return self.value is not None


class Tree : 

    def __init__(self, max_depth = 6, min_sample_per_split=2) : 

        self.max_depth = max_depth 
        self.min_sample_per_split = min_sample_per_split
        self.root = None

    def _split(self, X, thresh) : 
        """
        Split on X based on thresh

        args : 
            X : axis 
            thresh : value of split
        return : 
            left_idx : where X <= thresh
            right_idx : where X > thresh
        """
        left_idx = np.argwhere(X <= thresh).flatten()
        right_idx = np.argwhere(X > thresh).flatten()

        return (left_idx, right_idx)   

    def _entropyImpurity(self, Y) : 
        """
        Calculate the entropy impurity on Y
        """
        
        _, c = np.unique(Y, return_counts=True)
        a =  c / Y.shape[0]
        entropy = - np.sum(a * np.log2(a + 1e-9))
        return entropy

    def _infGain(self, X, y, thresh) : 
        """
        Calculate the information gain. 
        """
        my_loss = self._entropyImpurity(y)


        l_idx, r_idx = self._split(X, thresh)

        n, n_l, n_r = len(y), len(l_idx), len(r_idx)

        chid_loss = (n_l / n) * self._entropyImpurity(y[l_idx]) + (n_r / n) * self._entropyImpurity(y[r_idx])

        return my_loss - chid_loss
    
    def _bestSplit(self, X, y, features) :

        split = {
            'score' : -1, 
            'feature' : None, 
            'threshold' : None
        }

        for feature in features : 
            X_feat = X[:, feature]
            thresholds = np.unique(X_feat)
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

        self.n_samples, self.n_features = X.shape
        self.n_classes = len(np.bincount(y))


        # base case
        if self._finished(depth) :
            if np.unique(y).size == 0: 
                return  
            return Node(value = np.argmax(np.unique(y)))
        
        
        # At the moment we select random features but we can choose the one with the smallest entropy as well
        feats = np.random.choice(self.n_features, self.n_features, replace=False)

        best_feat, best_thresh = self._bestSplit(X, y, feats)

        #recursive step
        l_idx, r_idx = self._split( X[:, best_feat], best_thresh)
        l = self._build( X[l_idx, :], y[l_idx], depth + 1)
        r = self._build(X[r_idx, :], y[r_idx], depth + 1)

        return Node(best_feat, best_thresh, l, r)
    
    def fit(self, X, y): 
        self.root = self._build(X,y)
    

    def _traverse(self, x, node): 
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.thresh:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X) : 
        predictions = [self._traverse(x, self.root) for x in X]
        return predictions


from sklearn.model_selection import train_test_split
from sklearn import tree

X = dF.drop(['Y'], axis = 1).to_numpy() 
y = dF.Y.to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.9)

mytree = Tree()
mytree.fit(X_train, Y_train)

pred = mytree.predict(X_test)

print(sum(pred == Y_test) / Y_test.shape[0])

clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=6, min_samples_split=2).fit(X_train, Y_train)

pred = clf.predict(X_test)
print(sum(pred == Y_test) / Y_test.shape[0])