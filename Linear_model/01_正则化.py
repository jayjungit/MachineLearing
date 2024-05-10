import numpy as np
from sklearn.linear_model import Ridge, SGDRegressor, Lasso, ElasticNet

# L2正则化和普通线性回归系数对比：
X = 2 * np.random.randn(100, 5)
w = np.random.randint(1, 10, size=(5, 1))
b = np.random.randint(1, 10, size=1)
y = X.dot(w) + b + np.random.randn(100, 1)

# print('原始方程的斜率为：', w.ravel())
# print('原始方程的截距为：', b)

# 岭回归求解
ride = Ridge(alpha=1, solver='sag')
ride.fit(X, y)
# print('岭回归求解的斜率：', ride.coef_)
# print('岭回归求解的截距：', ride.intercept_)


# 线性回归梯度下降求解
sgd = SGDRegressor(penalty='l2', alpha=0, l1_ratio=0)
sgd.fit(X, y.reshape(-1, ))
# print('随机梯度下降求解的斜率为：', sgd.coef_)
# print('随机梯度下降求解的截距为：', sgd.intercept_)


# L1正则化和普通线性回归系数对比：
X = 2 * np.random.randn(100, 20)
w = np.random.randn(20, 1)
b = np.random.randint(1, 10, size=1)
y = X.dot(w) + b + np.random.randn(100, 1)

# print('原始方程的斜率为：', w.ravel())
# print('原始方程的截距为：', w.ravel())

# 套索回归的求解
lasso = Lasso(alpha=0.5)
lasso.fit(X, y)
# print('套索回归的斜率为：', lasso.coef_)
# print('套索回归的截距为：', lasso.intercept_)

# 线性回归梯度下降求解
sgd = SGDRegressor(penalty='l2', alpha=0, l1_ratio=0)
sgd.fit(X, y.reshape(-1, ))
# print('随机梯度下降的斜率为：', sgd.coef_)
# print('随机梯度下降的截距为：', sgd.intercept_)


# 弹性网络回归
"""Elastic-Net 回归，即岭回归和Lasso技术的混合。弹性网络是一种使用 L1， L2 范数作为先验正则项训练的线性回归模型。 这种组合允许学习到一个只有少
量参数是非零稀疏的模型，就像 Lasso 一样，但是它仍然保持一些像 Ridge 的正则性质。我们可利用 l1_ratio 参数控制 L1 和 L2 的凸组合。"""
enet = ElasticNet(alpha=1, l1_ratio=0.7)
enet.fit(X, y)
print('弹性网络的斜率为', enet.coef_)
print('弹性网络的截距为', enet.intercept_)
