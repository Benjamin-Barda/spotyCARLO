from kmean_class import KMeans_plus_plus as Kmm
import numpy as np 
import pandas as pd

dF = pd.read_csv('data//csvs//dataframeV2.csv', index_col=0)
dF = dF.drop(['id', 'uri'], axis = 1)
dF.head()

dF.label = pd.Categorical(dF.label)
dF['Y'] = dF.label.cat.codes
dF = dF.drop(['label'], axis = 1)

# Prepare the dataset for the decision tree
y = dF.Y 
X = dF.drop(["Y"], axis=1)


k = Kmm(K = 4, init=None)
k.fit_predict(X.to_numpy())