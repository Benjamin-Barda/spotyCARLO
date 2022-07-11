import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression():

    def __init__(self, basisFunc = lambda x:x):
        self.basis = basisFunc

    def fit(self,X,Y):
        self.X_copy = X.copy()
        self.X = X
        self.Y = Y
        ones = np.ones(self.X.shape[0],np.int64)
        self.X.insert(0,'intercept',ones)             #building the design matrix
        self.theta = ((np.linalg.inv(self.X.T @ self.X)) @ (self.X.T) @ self.Y).to_numpy()       #closed form of linear regression
    
    def predict(self, test ):
        test = np.array(test)
        test = test.reshape((test.shape[0],1))
        test = np.insert(test,0,1,axis=1)
        return test.dot(self.theta)
    
    def plot2D(self):
        if len(self.theta) != 2:
            print("Can't plot with this number of parameters")
            return None
        plt.scatter(self.X_copy,self.Y)
        a = np.linspace(0,55,100)
        def line(A):
            return self.theta[0] + self.theta[1]*A
        plt.plot(a,line(a),color='green')
        plt.show()

        

dF = pd.read_csv('data//csvs//LRtest.csv', index_col=0)
x = dF.drop('happiness',axis = 1)
y = dF['happiness']
LR = LinearRegression()
LR.fit(x,y)
LR.plot2D()
test = [25,37,52]
res = LR.predict(test)
print(res)
