import pandas as pd
import scipy as sp
from sklearn.metrics import log_loss
import pylab as pl
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split

data_frame=pd.read_csv('data.csv')

X = data_frame[['speed_p','speed_r','speed_d','distance_d_p','distance_d_r','angle_d','possTimePre','possessionTime']]
Y = data_frame[['flag']]

#用pandas加载数据.csv文件，然后用train_test_split分成训练集（75%）和测试集（25%）：
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X,Y,random_state=1)

#LogisticRegression同样实现了fit()和predict()方法
X_train=X_train_raw
X_test=X_test_raw
classifier=LogisticRegression(C=1)
classifier.fit(X_train,y_train)

predictions=classifier.predict(X_test)
probabilities = classifier.predict_proba(X_test)
print(probabilities)
y_test1=y_test.values.T[0]

def rmse(y_test, y):
	return sp.sqrt(sp.mean((y_test - y) ** 2))

print(rmse(predictions,y_test1))
print(log_loss(y_test1,predictions))
