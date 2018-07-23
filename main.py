import pandas as pd
data_frame=pd.read_csv('data.csv')

# print(data_frame.head())

X = data_frame[['speed_p','speed_r','speed_d','distance_d_p','distance_d_r','angle_d','possTimePre','possessionTime']]
Y = data_frame[['flag']]

import pylab as pl
# data_frame.hist()
# pl.show()

# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split
#用pandas加载数据.csv文件，然后用train_test_split分成训练集（75%）和测试集（25%）：
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X,Y)


# print(data_frame.describe())
print(X_test_raw.shape)
print(X_train_raw.shape)
print(y_test.shape)
print(y_train.shape)

#我们建一个TfidfVectorizer实例来计算TF-IDF权重：
# vectorizer=TfidfVectorizer()
# X_train=vectorizer.fit_transform(X_train_raw)
# X_test=vectorizer.transform(X_test_raw)
#LogisticRegression同样实现了fit()和predict()方法
X_train=X_train_raw
X_test=X_test_raw
classifier=LogisticRegression()
classifier.fit(X_train,y_train)

print(classifier.coef_)

# predictions=classifier.predict(X_test)
predictions=classifier.predict(X_test)
probabilities=classifier.predict_proba(X_test)
# probabilities = prediction

score = classifier.score(X_test, y_test)

params = classifier.get_params();
print(params)

print("score", score)

print("probabilities",probabilities)
print(y_test.shape)

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.scatter(y_test, predictions)
# ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
# ax.set_xlabel('Measured')
# ax.set_ylabel('predictions')
# plt.show()

print(predictions[1])
print(y_test.values[1][0])

print(y_test.shape)

count = 0
correct = 0
for i in range(len(y_test)):
	if y_test[i][0] == predictions[i]:
		correct += 1
	count += 1
	pass

print(correct, count)
# print(predictions)y_test
# print(y_test)
# for i ,prediction in enumerate(predictions[-5:]):
#     print ('预测类型：%s.信息：%s' %(prediction,X_test_raw.iloc[i]) )
