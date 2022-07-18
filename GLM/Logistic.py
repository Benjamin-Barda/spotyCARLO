import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


dF = pd.read_csv('data//csvs/dataframeV2.csv', index_col=0)
#x = dF[['danceability','energy','']]
#x = dF.drop(['popularity','id','uri','label'],axis=1)
x = dF[['energy']]
y = dF[['label']].to_numpy().ravel()

ly = LabelEncoder()
y = ly.fit_transform(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

logreg = LogisticRegression(solver = 'newton-cg',multi_class='auto',max_iter=100000,n_jobs=-1)
logreg.fit(x_train,y_train)

predictions = logreg.predict(x_test)

y = np.where(y == 2,1,0)

a = np.linspace(0,1,70)[:,np.newaxis]
probs = logreg.predict_proba(a)[:,2]
acc = accuracy_score(y_test,predictions)
plt.scatter(x[::5],y[::5],edgecolors='blue',c='aqua')
plt.plot(a,probs,lw=2,c='orangered')
plt.show()
print(acc)
