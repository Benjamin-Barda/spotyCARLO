import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression():

    def __init__(self, basisFunc = lambda x:x):
        self.basis = basisFunc

    def fit(self,X,Y):
        self.X = X
        self.Y = Y
        ones = np.ones(self.X.shape[0],np.int64)
        self.X.insert(0,'intercept',ones)             #building the design matrix
        self.theta = ((np.linalg.inv(self.X.T @ self.X)) @ (self.X.T) @ self.Y).to_numpy()       #closed form of linear regression
    
    def predict(self, X ):
        X = np.array(X).T
        #print(X)
        X = np.insert(X,0,1,axis=1)
        print(X)
        return X.dot(self.theta)
        



dF = pd.read_csv('data//csvs//LRtest.csv', index_col=0)
x = dF.drop('happiness',axis = 1)
y = dF['happiness']
LR = LinearRegression()
LR.fit(x,y)
plt.scatter(dF['money'],y)
a = np.linspace(0,55,100)
def retta(x):
    return 0.8146 + 1.9842*x
plt.plot(a,retta(a))
res = LR.predict([[25,37]])
print(res)
plt.scatter([25,37],res,color = 'red')
plt.show()