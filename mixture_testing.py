import numpy as np 
import pandas as pd
from MoG.mixturemodel import mixtureModel
from sklearn.model_selection import train_test_split


dF = pd.read_csv('data//csvs//dataframeV1.csv', index_col=0)
dF = dF.drop(['id', 'uri'], axis = 1)
dF.label = pd.Categorical(dF.label)
dF['Y'] = dF.label.cat.codes
dF = dF.drop(['label'], axis = 1)

# Prepare the dataset for the decision tree
y = dF.Y 
X = dF.drop(["Y"], axis=1)

# Split in train and validation
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=43, train_size=.75)



m = mixtureModel(4)
m.fit(X.to_numpy())

