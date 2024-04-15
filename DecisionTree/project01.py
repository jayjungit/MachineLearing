import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn import datasets


# 查看本电脑的字体
fm = font_manager.fontManager
font_list = [font.name for font in fm.ttflist]
# print(font_list)

# # 画图设置中文字体
# plt.rcParams['font.family'] = 'KaiTi'
#
# # #  鸢尾花数据集进行fl
# # X, y = datasets.load_iris(return_X_y=True) # return_X_y=True  X，y 分开
# # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=256)
# # model = DecisionTreeClassifier(criterion="entropy")
# # model.fit(X_train, y_train)
# #
# # y_ = model.predict(X_test)
# # print("预测的类别：", y_)
# # print("真实的类别：", y_test)
# # print("准确率为：", model.score(X_test, y_test))
#
#
# # 剪枝叶操作
# iris = datasets.load_iris()
# # print(iris.keys())
# X, y = iris["data"], iris["target"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=256)
#
#
# # max_depth调整树深度：剪枝操作
# # max_depth默认，深度最大，延伸到将数据完全划分开为止。
# # min_impurity_decrease（节点划分最小不纯度）如果某节点的不纯度(基尼系数，信息增益，均方差)小于这个阈值，则该节点不再生成子节点
# # max_depth（决策树最大深度）；min_samples_split（内部节点再划分所需最小样本数）
# # min_samples_leaf（叶子节点最少样本数）；max_leaf_nodes（最大叶子节点数）
# model = DecisionTreeClassifier(max_depth=3, criterion='gini', min_impurity_decrease=0.2)
# model.fit(X_train, y_train)
# y_ = model.predict(X_test)
# print("预测值为：", y_)
# print("正确值为：", y_test)
# print("准确率为： ", model.score(X_test, y_test))


# 找出最优的超参数
depth = np.arange(1, 20)
print(depth)
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=32)
err = []

for i in depth:
    model = DecisionTreeClassifier(max_depth=i, criterion='entropy')
    model.fit(X_train, y_train)
    y_ = model.predict(X_test)
    err.append(1 - model.score(X_test, y_test))
print(err)

plt.rcParams['font.family'] = 'STKaiTi'

plt.plot(depth, err, 'ro-')
plt.xlabel("树的深度")
plt.ylabel("错误率")
plt.title("筛选合适的决策树的深度")
plt.grid()
plt.savefig("筛选数深度超参数.png")
plt.show()