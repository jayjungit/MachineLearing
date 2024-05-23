from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

# 二分类
X, y = datasets.load_iris(return_X_y=True)
cond = y != 2
X = X[cond]
y = y[cond]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))


# 多分类
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
model = LogisticRegression(multi_class='ovr')
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

# 多分类
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
model = LogisticRegression(multi_class='multinomial')
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
