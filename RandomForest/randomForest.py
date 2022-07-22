from lib2to3.pytree import Base
import numpy as np 
import pandas as pd
from DecisionTree.tree import Tree
from sklearn.base import BaseEstimator


class Forest(BaseEstimator) : 

    def __init__(self, max_trees = 10, max_depth = 6, min_sample_split = 2, random_seed = None) :

        self.rs = random_seed

        self.B = max_trees
        self.max_depth = max_depth 
        self.min_sample_split = min_sample_split 
        self.forest = list()
        self._fit = False

    def fit(self, X, y): 

        con = pd.concat([X, y.to_frame()], axis = 1)

        # Bootstrap
        for b in range(self.B): 
            df_boot = con.sample(n = len(con), replace = True, random_state=self.rs)
            y_boot = df_boot.Y.to_numpy()
            x_boot = df_boot.drop(['Y'], axis = 1).to_numpy()
            t_boot = Tree(inForest=True, random_seed=self.rs)
            t_boot.fit(x_boot, y_boot)
            self.forest.append(t_boot)

        
        self._fit = True

    def predict(self, X) :

        if type(X).__module__ != np.__name__: 
            X = X.to_numpy()


        final = list()

        if not self._fit : 
            print("Forest not yet populated")
            return -1

        # Shape = (num_trees, num_samples)
        preds = np.zeros((self.B, len(X)), dtype=np.int64)

        for b,tree in enumerate(self.forest) :
            pred = np.asarray(tree.predict(X), dtype=np.int8)
            preds[b, :] = pred
        

        for i in range(len(X)) : 
            temp = np.argmax(np.bincount(preds[:,i]))
            final.append(temp)

        return final 
    
    def score(self, X, y): 
        preds = self.predict(X)

        return sum(preds == y) / len(preds)

    def get_params(self,deep=True) : 
        return {
            'max_trees' : self.B,
            'max_depth' : self.max_depth, 
            'min_sample_split' : self.min_sample_split 
            }
    
    def set_params(self, **params):
        for parameter, value in params.items() : 
            setattr(self, parameter, value)


        