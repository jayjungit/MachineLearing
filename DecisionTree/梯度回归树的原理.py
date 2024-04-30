import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


X = np.array([[600, 0.8], [800, 1.2], [1500, 10], [2500, 3]])
y = np.array([14, 16, 24, 26])

lr = 0.1
gbdt = GradientBoostingRegressor(n_estimators=3, learning_rate=lr)
gbdt.fit(X, y)
y_ = gbdt.predict(X)
# print(y_)


residual0 = y - y.mean()

# 实现第一课树
residual1 = residual0 - residual0 * 0.1
print(residual1)

# 第二棵树
residual2 = residual1 - 0.1 * residual1

# 第三颗树
residual3 = residual2 - 0.1 * residual2
print(residual3)

# 预测结果
pred = y - residual3
print(pred)


# 第一棵树最佳分裂条件
lower_mse = ((y - y.mean()) ** 2).mean()
best_split = {}

for index in range(2):
    for i in range(3):
        t = X[:, index].copy()
        t.sort()
        split = t[i: i + 2].mean()
        cond = X[:, index] <= split
        mse1 = ((y[cond] - y[cond].mean()) ** 2).mean()
        mse2 = ((y[~cond] - y[~cond].mean()) ** 2).mean()
        p1 = cond.sum() / cond.size
        mse = p1 * mse1 + (1 - p1) * mse2
        if mse < lower_mse:
            lower_mse = mse
            best_split.clear()
            best_split['第%d'%(index)] = split
        elif mse == lower_mse:
            best_split['第%d'%(index)] = split

print("最佳分裂点：", best_split)


# 第二棵树最佳分裂点
lower_mse = round(((residual0 - residual0.mean())**2).mean(),3)
print('未分裂均方误差是：',lower_mse)
best_split = {}
for index in range(2):
    for i in range(3):
        t = X[:,index].copy()
        t.sort()
        split = t[i:i + 2].mean()
        cond = X[:,index] <= split
        mse1 = round(((residual1[cond] - residual1[cond].mean())**2).mean(),3)
        mse2 = round(((residual1[~cond] - residual1[~cond].mean())**2).mean(),3)
        p1 = cond.sum()/cond.size
        mse = round(mse1 * p1 + mse2 * (1- p1),3)
        print('第%d列' % (index),'裂分条件是：',split,'均方误差是：',mse1,mse2,mse)
        if mse < lower_mse:
            best_split.clear()
            lower_mse = mse
            best_split['第%d列'%(index)] = split
        elif mse == lower_mse:
            best_split['第%d列'%(index)] = split
print('最佳分裂条件是：',best_split)