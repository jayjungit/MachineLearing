import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# movie = pd.read_csv('../data/IMDB-Movie-Data.csv')
# print(np.any(pd.isnull(movie)))
# 查看空值的数量
# print(movie.isnull().sum())

# print(movie['Revenue (Millions)'].mean())

# 用每个column的平均值填充空值
# print(movie.columns)
# for i in movie.columns:
#     if np.any(pd.isnull(movie[i])):
#         print(i)
#         movie[i].fillna(movie[i].mean(), inplace=True)


# 删除有空的数据
# movie.dropna(inplace=True)
# print(np.any(pd.isnull(movie)))


# data = pd.read_csv('../data/stock_day.csv')
# # print(data.head())
# # print(data.columns)
# p_change = data['p_change']
# # print(p_change.head())
#
# #  将数据按分位数进行分箱（分段）
# qcut = pd.qcut(p_change, 10)
# # print(qcut.value_counts())
# # print(np.min(p_change))
#
# # 按区间进行分箱
# bins = [-100, -7, -5, -3, 0, 3, 5, 7, 100]
# p_counts = pd.cut(p_change, bins)
# print(p_counts.value_counts())
#
# dummies = pd.get_dummies(p_counts, prefix='rise')
# # print(dummies)
# # 按列进行合并
# data = pd.concat([data, dummies], axis=1)
# print(data)

# left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
#                      'key2': ['K0', 'K1', 'K0', 'K1'],
#                      'A': ['A0', 'A1', 'A2', 'A3'],
#                      'B': ['B0', 'B1', 'B2', 'B3']})
#
# right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
#                       'key2': ['K0', 'K0', 'K0', 'K0'],
#                       'C': ['C0', 'C1', 'C2', 'C3'],
#                       'D': ['D0', 'D1', 'D2', 'D3']})
#
# print(left)
# print("==============================")
# print(right)
# print("==============================")
#
# # 内连接
# merge1 = pd.merge(left, right, on=['key1', 'key2'], how='inner')
# print(merge1)
#
# # 左连接
# merge2 = pd.merge(left, right, on=['key1', 'key2'], how='left')
# print(merge2)
#
# # 右连接
# merge3 = pd.merge(left, right, on=['key1', 'key2'], how='right')
# print(merge3)
#
# # 外连接
# merge4 = pd.merge(left, right, on=['key1', 'key2'], how='outer')
# print(merge4)


# print(data.index)
# time = pd.to_datetime(data.index)
# # print(time.weekday)
# data['week'] = time.weekday
# # print(data.head())
# # print(time.day)
#
# data['p_n'] = np.where(data['p_change'] > 0, 1, 0)
# # print(data)
#
#
# count = pd.crosstab(data['week'], data['p_n'])
# # print(count)
#
# sum = count.sum(axis=1).astype(np.float32)
# # print(sum)
#
# ret = count.div(sum, axis=0)
# # print(ret)
# ret.plot(kind='bar', stacked=True)
# # plt.show()
# print(data.pivot_table(['p_n'], index='week'))

# 分组聚合
col = pd.DataFrame(
    {'color': ['white', 'red', 'green', 'red', 'green'], 'object': ['pen', 'pencil', 'pencil', 'ashtray', 'pen'],
     'price1': [5.56, 4.20, 1.30, 0.56, 2.75], 'price2': [4.75, 4.12, 1.60, 0.75, 3.15]})

# print(col)
# group1 = col.groupby(['color'])['price1'].mean()
# print(group1)
#
# group2 = col.groupby(['color'], as_index=False)["price2"].mean()
# print(group2)


data = pd.read_csv("../data/directory.csv")
# print(data)

group = data.groupby(['Country']).count()
# print(group)
# print(data['Brand'].unique())
group['Brand'].plot(kind='bar', figsize=(20, 8))
# plt.show()

group2 = data.groupby(['Country', 'Brand']).count()
print(group2)