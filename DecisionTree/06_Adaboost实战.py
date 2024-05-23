import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt



# X, y = datasets.load_breast_cancer(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# # 使用决策树
# model1 = DecisionTreeClassifier()
# model1.fit(X_train, y_train)
# y1_pred = model1.predict(X_test)
# print("决策树得分数：", accuracy_score(y_test, y1_pred))
#
#
# # 使用随机森林
# model2 = RandomForestClassifier(n_estimators=500)
# model2.fit(X_train, y_train)
# y2_pred = model2.predict(X_test)
# print("随机森林得得分：", accuracy_score(y_test, y2_pred))
#
#
# # 使用Adaboost
# model3 = AdaBoostClassifier(n_estimators=500)
# model3.fit(X_train, y_train)
# y3_pred = model3.predict(X_test)
# print('Adaboost的得分：', accuracy_score(y_test, y3_pred))


# 手写数字
data = pd.read_csv('../data/digits.csv')
data = data.take(np.random.randint(0, 42000, 5000))
X = data.iloc[:, 1:]
y = data['label']
print(y.shape)
X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.2)

# 打印一张图片
# image = X_train.iloc[:1, :]
# plt.imshow(image.values.reshape(28, 28))
# plt.show()
# print(X_test.iloc[:1, :])


# model = AdaBoostClassifier(n_estimators=100)
# model.fit(X_train, y_train)
# y_ = model.predict(X_test)
# print("Adaboost得分为：", accuracy_score(y_test, y_))


model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
y_ = model.predict(X_test)
print("逻辑斯蒂回归得得分为:", accuracy_score(y_test, y_))


plt.figure(figsize=(5*2, 10*2))
plt.rcParams['font.family'] = 'KaiTi'
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(X_test.iloc[i].values.reshape(28, 28))
    plt.title("预测值是：%d" %y_[i], fontsize=10)
    plt.axis('off')
plt.show()



