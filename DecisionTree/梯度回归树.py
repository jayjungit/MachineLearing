import numpy as np
from sklearn.ensemble import GradientBoostingClassifier


X = np.arange(1, 11).reshape(-1, 1)
y = np.array([0, 0, 0, 1, 1] * 2)
model = GradientBoostingClassifier(n_estimators=3, max_depth=1, learning_rate=0.1)
model.fit(X, y)
y_ = model.predict(X)
# print(y_)
proba = model.predict_proba(X)
# print("概率为：\n", proba)

# 计数初始值
F0 = np.log(y.sum() / (1 - y).sum())
F0 = np.array([F0] * 10)
# print(F0)
residual0 = y - (1 / (1 + np.exp(-F0)))
# print(residual0)

# 第一棵树计算过程
lower_mse = ((residual0 - residual0.mean()) ** 2).mean()
best_split = {}
for i in range(1, 10):
    if i == 9:
        mse = ((residual0 - residual0.mean()) ** 2).mean()
    else:
        left_mse = ((residual0[: i + 1] - residual0[:i+1].mean()) ** 2).mean()
        right_mse = ((residual0[i+1:] - residual0[i+1:].mean()) ** 2).mean()
        mse = (i + 1)/10 * left_mse + (10 - 1 - i)/10 * right_mse
        if mse < lower_mse:
            lower_mse = mse
            best_split.clear()
            best_split["X[0]<="] = X[i:i+2].mean()

print("最佳分裂点：", best_split)
print("最小mse：", np.round(lower_mse, 2))

gamma1 = residual0[:8].sum() / ((y[:8] - residual0[:8]) * (1 - y[:8] + residual0[:8])).sum()
gamma2 = residual0[8:].sum() / ((y[8:] - residual0[8:]) * (1 - y[8:] + residual0[8:])).sum()

gamma = np.array(([gamma1] * 8) + ([gamma2] * 2))
print("gamma1", gamma1)
print("gamma2", gamma2)
F1 = F0 + gamma * 0.1
# F = np.array([F1] * 10)
print("F1:", F1)

# 第 2 3 4 ... 棵树同理
# 计算概率
p = 1 / (1 + np.exp(-F1))
print(p)
