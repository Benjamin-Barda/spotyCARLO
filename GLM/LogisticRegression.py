import pandas as pd
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


dF = pd.read_csv('data//csvs/dataframeV2.csv', index_col=0)
x = dF.drop(['id','uri','label'],axis=1)
y = dF[['label']].to_numpy().ravel()

ly = LabelEncoder()
y = ly.fit_transform(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.22)


logreg = LogisticRegression(solver = 'newton-cg',multi_class='auto',max_iter=100000,n_jobs=-1)
logreg.fit(x_train,y_train)

y_pred = logreg.predict(x_test)

acc1 = accuracy_score(y_test,y_pred)
print(acc1)