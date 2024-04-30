# bagging
# 自建集成算法
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
warnings.filterwarnings('ignore')


X, y = datasets.load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1024)

# KNN单一算法
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print("KNN单一算法准确率为：", knn.score(X_test, y_test))

# KNN集成算法
knn_big = BaggingClassifier(base_estimator=knn, n_estimators=100, max_samples=0.8, max_features=0.7)
knn_big.fit(X_train, y_train)
print("Big_knn集成算法分数：", knn_big.score(X_test, y_test))


# 逻辑斯蒂回顾集成
# lr = LogisticRegression()
# lr_big = BaggingClassifier(base_estimator=LogisticRegression(), n_estimators=500, max_features=0.8, max_samples=0.5)
# lr.fit(X_train, y_train)
# lr_big.fit(X_train, y_train)
# print('逻辑斯蒂单一算法分数：', lr.score(X_test, y_test))
# print('逻辑斯集成算法分数：', lr_big.score(X_test, y_test))


# 决策树自建集成算法
clf = DecisionTreeClassifier()
clf_big = BaggingClassifier(base_estimator=clf, n_estimators=100, max_features=1.0, max_samples=0.5)
clf.fit(X_train, y_train)
clf_big.fit(X_train, y_train)
print('决策树单一算法分数：', clf.score(X_test, y_test))
print('决策树集成算法分数：', clf_big.score(X_test, y_test))


