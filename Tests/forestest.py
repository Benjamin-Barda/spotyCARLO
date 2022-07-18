import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from DecisionTree.tree import Tree
from RandomForest.randomForest import Forest


dF = pd.read_csv('data//csvs//dataframeV1.csv', index_col=0)
dF = dF.drop(['id', 'uri'], axis = 1)
dF.head()

dF.label = pd.Categorical(dF.label)
dF['Y'] = dF.label.cat.codes
dF = dF.drop(['label'], axis = 1)

# Prepare the dataset for the decision tree
y = dF.Y 
X = dF.drop(["Y"], axis=1)

print(X.shape)

#Split in train and validation
x_train, x_test, y_train, y_test = train_test_split(X, y,  train_size=.75)

f = Forest(32, max_depth=5) 
f.fit(x_train, y_train)

p = f.predict(x_test.to_numpy())
print(sum(p == y_test) / len(y_test))
print(x_test)
