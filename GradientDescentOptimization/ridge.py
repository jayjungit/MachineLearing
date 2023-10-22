import numpy as np
from sklearn.linear_model import Ridge, SGDRegressor


X = 2 * np.random.rand(100, 5)
w = np.random.randint(1, 10, size=(5, 1))
b = np.random.randint(1, 10, size=1)

y = X.dot(w) + b + np.random.rand(100, 1)

print('原方程的斜率： ', w.ravel())
print('原方程的截距：', b)

ridge = Ridge(alpha=1,solver='sag')
ridge.fit(X, y)
print("ridge求解的斜率：", ridge.coef_)
print("ridge求解的截距：", ridge.intercept_)

sgd = SGDRegressor(penalty='l2', alpha=0, l1_ratio=0)
sgd.fit(X, y)
print('随机梯度下降求解的斜率：', sgd.coef_)
print('随机梯度下降求解的截距：', sgd.intercept_)




