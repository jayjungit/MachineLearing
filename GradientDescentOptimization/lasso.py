import numpy as np
from sklearn.linear_model import Lasso, SGDRegressor


X = 2 * np.random.rand(100, 20)
w = np.random.randn(20, 1)
b = np.random.randint(1, 10, size=1)
y = X.dot(w) + b + np.random.randn(100, 1)

print('原来方程的斜率：', w.ravel())
print('原来过程的截距：', b)

lasso = Lasso(alpha=0.5)
lasso.fit(X, y)
print('lasso求解的斜率：', lasso.coef_)
print('lasso求解的截距：', lasso.intercept_)

sgd = SGDRegressor(penalty='l2', alpha=0, l1_ratio=0)
sgd.fit(X, y)
print('随机梯度下降求解的斜率：', sgd.coef_)
print('随机梯度下降求解的截距：', sgd.intercept_)
