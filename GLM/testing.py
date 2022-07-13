import numpy as np
import pandas as pd

dF = pd.read_csv('data//csvs/polytest.csv', index_col=0)
x = dF.drop('Distance',axis = 1)
#print(x)
ones = np.ones(x.shape[0],np.int64)
x.insert(0,'intercept',ones)     
#print(x)
#x=x.assign(square=lambda z:z.iloc[:,1]**2)
#print(x)

theta = np.array([169.5,1.555,3.2])
xmin,xmax = 10,64
a = np.linspace(xmin,xmax,10)[:,np.newaxis]
ones = np.ones(10,np.int64)[:,np.newaxis]
arr = np.concatenate((ones,a),axis=1)
arr = np.concatenate((arr,a**2),axis=1)
print(arr)
def line(A):
   return arr.dot(theta)
print(line(arr))
