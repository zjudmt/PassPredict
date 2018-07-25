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

# print(data_frame.describe())
# print(X_test_raw.shape)
# print(X_train_raw.shape)
# print(y_test.shape)
# print(y_train.shape)

#LogisticRegression同样实现了fit()和predict()方法
X_train=X_train_raw
X_test=X_test_raw
classifier=LogisticRegression()
classifier.fit(X_train,y_train)

print(classifier.coef_)

predictions=classifier.predict(X_test)
# predictions = prediction

print(predictions)
print(y_test.shape)

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.scatter(y_test, predictions)
# ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
# ax.set_xlabel('Measured')
# ax.set_ylabel('predictions')
# plt.show()
y_test1=y_test.values.T[0]
print(predictions[1])
print(y_test.values[1][0])

print(y_test1)
print(predictions)


#print(predictions)
def rmse(y_test, y):
	return sp.sqrt(sp.mean((y_test - y) ** 2))

# def log_loss(true_y,pred_h):
# 	return -np.mean(true_y*np.log(pred_h)+(1-true_y)*np.log(1-pred_h))

print(rmse(predictions,y_test1))
print(log_loss(y_test1,predictions))
# print(predictions)y_test
# print(y_test)
# for i ,prediction in enumerate(predictions[-5:]):
#     print ('预测类型：%s.信息：%s' %(prediction,X_test_raw.iloc[i]) )
