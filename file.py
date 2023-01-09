import pandas as pd
data=pd.read_csv("credit.csv",encoding='windows-1252').dropna()
print(data)
X=data.drop(['Class'],axis=1)
print(X)
Y=data['Class']
print(Y)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30) 
print(x_train)
print(x_test)
print(y_train)
print(y_test)
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score
acc1=accuracy_score(y_pred,y_test)
print(acc1)
