import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression():

    def __init__(self, degree = 1):
        self.degree = degree

    def fit(self,X,Y):
        self.X_copy = X.copy()
        self.X = X
        self.Y = Y
        self.X.insert(0,'intercept',1)             # building the design matrix adding column of ones (1,1,1,1,...,1)
        if self.degree > 1:                        # in case of polynomial regression
            if self.X.shape[1] != 2:
                print('ERROR: trying to do polynomial regression with multiple dimensions')
                return None                           
            for i in range(2,self.degree + 1):                                  # "basis function" to lift up the features 
                self.X.insert(i,x.columns.values[1]+f'^{i}',x.iloc[:,1]**i)     # create columns of powers of the first feature up to the degree selected
        self.theta = ((np.linalg.inv(self.X.T @ self.X)) @ (self.X.T) @ self.Y).to_numpy()       #closed form of linear regression
        return self.theta
            
    def predict(self, test ):
        test = np.array(test)
        if len(test.shape) ==1:
            test  = test[:, np.newaxis]
        test2 = np.insert(test,0,1,axis=1)
        if self.degree > 1:
            for i in range(2,self.degree+1):
                test2 = np.concatenate((test2,test**i),axis=1)
        return test2.dot(self.theta)
    
    def plot2D(self):
        if len(self.theta) != 2 and self.degree == 1:
            print("Can't plot with this number of parameters")
            return None
        plt.scatter(self.X_copy,self.Y)
        xmin,xmax = int(self.X_copy.min()),int(self.X_copy.max())
        bounds = np.linspace(xmin,xmax,70)[:,np.newaxis]
        Xaxis = np.insert(bounds,0,1,axis=1)
        for i in range(2,self.degree+1):
            Xaxis = np.concatenate((Xaxis,bounds**i),axis=1)
        def line(A):
            return A.dot(self.theta)
        plt.plot(bounds,line(Xaxis),color='green')
        plt.show()

    def plot3D(self):
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        xx, yy = np.meshgrid(range(110), range(110))
        def plane(A,B):
            return self.theta[0] + self.theta[1]*A + self.theta[2]*B
        ax.scatter(self.X_copy.iloc[:,0], self.X_copy.iloc[:,1], self.Y, color='green')
        ax.plot_surface(xx, yy, plane(xx,yy), alpha=0.5)
        ax.set_xlabel(self.X_copy.columns.values[0])
        ax.set_ylabel(self.X_copy.columns.values[1])
        ax.set_zlabel(self.Y.name)
        plt.show()

        

dF = pd.read_csv('data//csvs/LRtest.csv', index_col=0)
x = dF.drop('happiness',axis = 1)
y = dF['happiness']
LR = LinearRegression()
LR.fit(x,y)
#test = [45,13]
#res = LR.predict(test)
LR.plot3D()
#print(res)
