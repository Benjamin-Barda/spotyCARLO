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
                raise Exception("ERROR: trying to do polynomial regression with multiple dimensions")                  
            for i in range(2,self.degree + 1):                                                          # "basis function" to lift up the features 
                self.X.insert(i,x.columns.values[1]+f'^{i}',x.iloc[:,1]**i)                             # create columns of powers of the first feature up to the degree selected
        self.theta = ((np.linalg.inv(self.X.T @ self.X)) @ (self.X.T) @ self.Y).to_numpy()              #closed form of linear regression
       
        InitVar = self.Y.var()                                  # Initial variance of the explained variable
        SSR = np.sum((self.predict(self.X_copy)-self.Y)**2)     # Sum of Square residual after regression
        self.R_squared = float(1 - (SSR / InitVar))                    # R2
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
    
    def plotModel(self,xlim = 1,ylim = 1):
        if len(self.theta) == 3 and self.degree == 1:
            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')
            xlim, ylim = np.linspace(0,xlim,70), np.linspace(0,ylim,70)
            xx, yy = np.meshgrid(xlim, ylim)
            def plane(A,B):
                return self.theta[0] + self.theta[1]*A + self.theta[2]*B
            ax.scatter(self.X_copy.iloc[:,0], self.X_copy.iloc[:,1], self.Y, color='green')
            ax.plot_surface(xx, yy, plane(xx,yy), alpha=0.5)
            ax.set_xlabel(self.X_copy.columns.values[0])
            ax.set_ylabel(self.X_copy.columns.values[1])
            ax.set_zlabel(self.Y.columns.values[0])
            plt.show()
        elif len(self.theta) == (1 + self.degree):
            plt.scatter(self.X_copy,self.Y,s=9.5,alpha=0.8)
            xmin,xmax = float(self.X_copy.min()),float(self.X_copy.max())
            bounds = np.linspace(xmin,xmax,70)[:,np.newaxis]
            Xaxis = np.insert(bounds,0,1,axis=1)
            for i in range(2,self.degree+1):
                Xaxis = np.concatenate((Xaxis,bounds**i),axis=1)
            def line(A):
                return A.dot(self.theta)
            plt.plot(bounds,line(Xaxis),color='red',lw=2)
            plt.show()
        else:
            raise Exception("ERROR: can't plot, too many dimensions")

dF = pd.read_csv('data//csvs/dataframeV2.csv', index_col=0)
x = dF.drop(['popularity','id','uri','label'],axis=1)
y = dF[['popularity']]
#print(y.mean())
LR = LinearRegression()
LR.fit(x,y)
#test = [[0.712,0.772,10,-3.024,0,0.346,0.0521,4.35e-06,0.0368,0.848,84.722,249480,4]]
#res = LR.predict(test)
#LR.plotModel()
print(LR.R_squared)
#print(res)
