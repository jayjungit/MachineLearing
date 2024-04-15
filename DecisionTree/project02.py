import numpy
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


# 决策回归树
X = np.linspace(0, 2*np.pi, 40).reshape(-1, 1)
y = np.c_[np.sin(X), np.cos(X)]
X_test = np.linspace(0, 2*np.pi, 187).reshape(-1, 1)

# plt.figure(figsize=(9, 9))
# plt.scatter(y[:, 0], y[:, 1])

model = DecisionTreeRegressor(max_depth=3)
model.fit(X, y)
y_ = model.predict(X_test)

# plt.scatter(y_[:, 0], y_[:, 1])
# plt.show()

mse = ((y - y.mean(axis=0))**2).mean()
print(mse)
# print(y[True])

# 寻找最佳分裂点
mse_result = []
split_result = {}
mse_lower = 0.5 # 未分裂时，mse，最大的
for i in range(len(X) - 1):
    split = round(X[i:i+2].mean(),3)
    cond = (X <= split).reshape(-1)
    left = y[cond]
    right = y[~cond]
    mse_left = ((left - left.mean(axis = 0))**2).mean()
    mse_right = ((right - right.mean(axis = 0))**2).mean()
    # 计算左右叶节点比例
    left_percent = cond.sum()/cond.size
    right_percent = 1 - left_percent
    # 计算整体
    mse = mse_left * left_percent + mse_right * right_percent
    mse_result.append(mse)
#     print(mse,left_percent,right_percent)
    if mse < mse_lower:
        split_result.clear()
        split_result[split] = mse
        mse_lower = mse # 更新记录最小mse变量
print('最佳裂分条件：',split_result)