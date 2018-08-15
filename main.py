import numpy as np
import pandas as pd
import scipy as sp
from sklearn.metrics import log_loss
import pylab as pl
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split

data_frame=pd.read_csv('data.csv')

X = data_frame[['speed_p','speed_r','speed_d','distance_d_p','distance_d_r','distance_d1_p','distance_d2_r' ,'angle_d','angle_d1_p','angle_d2_r' , 'possTimePre','possessionTime']]
Y = data_frame[['flag']]

#用pandas加载数据.csv文件，然后用train_test_split分成训练集（75%）和测试集（25%）：
X_train, X_test, y_train, y_test = train_test_split(X,Y.values.T[0],random_state=1)

#LogisticRegression同样实现了fit()和predict()方法
classifier=LogisticRegression()
classifier.fit(X_train,y_train)

predictions=classifier.predict(X_test)
probabilities = classifier.predict_proba(X_test)
score = classifier.score(X_test,y_test)

print("probabilities" ,probabilities)
print("R-requested", score)

def rmse(y_test, y):
	return sp.sqrt(sp.mean((y_test - y) ** 2))

# 均方误差及log-loss
print("rmse" ,rmse(predictions,y_test))
print("log_loss" ,log_loss(y_test,predictions))

# 线性组合系数数组
coef = classifier.coef_
print(coef)

# 相关系数矩阵
corr = data_frame.corr()
print(corr)
