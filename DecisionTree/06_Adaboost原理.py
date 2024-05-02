from sklearn.ensemble import AdaBoostClassifier
import numpy as np


X = np.arange(10).reshape(-1, 1)
y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])

model = AdaBoostClassifier(algorithm='SAMME', n_estimators=3)

model.fit(X, y)
# print('真实值：', y)
# print('预测值为：', model.predict(X))

# gini系数的计算
w1 = np.full(shape=10, fill_value=0.1)
cond = y == -1
p1 = w1[cond].sum()
cond = y == -1
p2 = w1[cond].sum()
gini = p1 * (1-p1) + p2 * (1 - p2)
# print(gini)

# 寻找最佳裂分条件
gini_result = []
best_split = {}
lower_gini = 1
for i in range(len(X) - 1):
    split = X[i: i+2].mean()
    cond = (X <= split).ravel()
    part1 = y[cond]
    part2 = y[~cond]
    gini1 = 0
    gini2 = 0
    for i in np.unique(y):
        p1 = (part1 == i).sum() / part1.size
        gini1 += p1 * (1 - p1)
        p2 = (part2 == i).sum() / part2.size
        gini2 += p2 * (1 - p2)
    part1_p = cond.sum() / cond.size
    part2_p = 1 - part1_p
    gini = part1_p * gini1 + part2_p * gini2
    gini_result.append(gini)
    if gini < lower_gini:
        lower_gini = gini
        best_split.clear()
        best_split['X[0] <='] = split

# print('gini的所有结果', gini_result)
# print('最佳分裂点为：', best_split)


# 计算误差率
y1_ = np.array([1 if X[i] <= 2.5 else -1 for i in range(10)])
# print(y_)
err = (y1_ != y).mean()
# print(err)

# 计算第一个弱分类器的权重
alpha1 = 1/2 * np.log((1-err)/err)
# print('第一个弱分类器的权重:', alpha1)

# 更新权重
w2 = w1 * np.exp(-alpha1 * y * y1_)
w2 = w2 / w2.sum()
# print("第一棵树结束更新权重:\n", w2)


# 第二棵树的计算
# 计算基尼系数
cond = y == -1
p1 = w2[cond].sum()
cond = y == 1
p2 = w2[cond].sum()
gini = 1 - p1**2 - p2**2
# print(gini)

# 最找第二课时的最佳分裂点
gini_result = []
best_split = {}
lower_gini = 1
for i in range(len(X) - 1):
    split = X[i: i+2].mean()
    cond = (X <= split).ravel()
    part1 = y[cond]
    part1_w2 = w2[cond] / w2[cond].sum()
    part2 = y[~cond]
    part2_w2 = w2[~cond] / w2[~cond].sum()
    gini1 = 1
    gini2 = 1
    for i in np.unique(y):
        cond1 = i == part1
        p1 = part1_w2[cond1].sum()
        gini1 -= p1 ** 2
        cond2 = i == part2
        p2 = part2_w2[cond2].sum()
        gini2 -= p2 ** 2
    part1_p = cond.sum() / cond.size
    part2_p = 1 - part1_p
    gini = part1_p * gini1 + part2_p * gini2
    # print(gini)
    gini_result.append(gini)
    if gini < lower_gini:
        lower_gini = gini
        best_split.clear()
        best_split['X[0] <= '] = split
# print("第二棵树的gini结果：", gini_result)
# print("第二棵树的最佳分裂点：", best_split)

# 计算误差率
y2_ = np.array([1 if X[i] <= 8.5 else -1 for i in range(10)])
err = ((y2_ != y)*w2).sum()
# print(err)
# 计算第一个弱分类器的权重
alpha2 = 1/2 * np.log((1-err)/err)
# print('第二个弱分类器的权重:', alpha1)

# 更新权重
w3 = w2 * np.exp(-alpha2 * y * y2_)
w3 = w3 / w3.sum()
# print("第二棵树结束更新权重:\n", w3)


# 第三棵树的计算
# 计算基尼系数
cond = y == -1
p1 = w3[cond].sum()
cond = y == 1
p2 = w3[cond].sum()
gini = 1 - p1**2 - p2**2
print(gini)

# 找第三棵树的最佳分裂点
gini_result = []
best_split = {}
lower_gini = 1
for i in range(len(X) - 1):
    split = X[i: i+2].mean()
    cond = (X <= split).ravel()
    part1 = y[cond]
    part1_w3 = w3[cond] / w3[cond].sum()
    part2 = y[~cond]
    part2_w3 = w3[~cond] / w3[~cond].sum()
    gini1 = 1
    gini2 = 1
    for i in np.unique(y):
        cond1 = i == part1
        p1 = part1_w3[cond1].sum()
        gini1 -= p1 ** 2
        cond2 = i == part2
        p2 = part2_w3[cond2].sum()
        gini2 -= p2 ** 2
    part1_p = cond.sum() / cond.size
    part2_p = 1 - part1_p
    gini = part1_p * gini1 + part2_p * gini2
    gini_result.append(gini)
    if gini < lower_gini:
        lower_gini = gini
        best_split.clear()
        best_split['X[0] <= '] = split
# print("第三棵树的gini结果：", gini_result)
# print("第三棵树的最佳分裂点：", best_split)

# 计算误差率
y3_ = np.array([-1 if X[i] <= 5.5 else 1 for i in range(10)])
err = ((y3_ != y)*w3).sum()
# print(err)
# 计算第三个弱分类器的权重
alpha3 = 1/2 * np.log((1-err)/err)
# print('第三弱分类器的权重:', alpha3)

# 更新权重
w4 = w3 * np.exp(-alpha3 * y * y3_)
w4 = w4 / w4.sum()
# print("第三棵树结束更新权重:\n", w4)


# 聚合弱分类器得结果
F = alpha1 * y1_ + alpha2 * y2_ + alpha3 * y3_
print('最终结果如下：\n',np.array([1 if i > 0 else -1 for i in F]))