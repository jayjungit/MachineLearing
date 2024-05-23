import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("../data/nba_2017_nba_players_with_salary.csv")
# print(data.head())
# print(data.info())
# print(data.describe())
# print(data.shape)

# 基于整数位置索引 (iloc) 和基于标签/条件 (loc)
data_cor = data.loc[:, ['RPM', 'AGE', 'SALARY_MILLIONS', 'ORB', 'DRB', 'TRB',
                        'AST', 'STL', 'BLK', 'TOV', 'PF', 'POINTS', 'GP', 'MPG', 'ORPM', 'DRPM']]
# print(data_cor)
# 获取两列数据之间的相关性
corr = data_cor.corr()
# print(corr)

# plt.figure(figsize=(20, 20), dpi=100)
# sns.heatmap(corr, square=True, linewidths=0.1, annot=True)

# plt.show()

# 基本数据排名分析
# 按照效率值排名
sort1 = data.loc[:, ['PLAYER', 'RPM', 'AGE']].sort_values(by='RPM', ascending=False)
# print(sort1)
# 按照薪水进行排名
sort2 = data.loc[:, ["PLAYER", "RPM", "AGE", "SALARY_MILLIONS"]].sort_values(by='SALARY_MILLIONS', ascending=False)
# print(sort2)

sns.set_style('darkgrid')


# seaborn.distplot() 是 Seaborn 用于绘制单变量分布图的函数
# plt.figure(figsize=(10, 10))
# plt.subplot(3, 1, 1)z
# sns.histplot(data['RPM'], kde=True)
# plt.ylabel('RPM')
#
# plt.subplot(3, 1, 2)
# sns.histplot(data['SALARY_MILLIONS'], kde=True)
# plt.ylabel('SALARY_MILLIONS')
#
# plt.subplot(3, 1, 3)
# sns.histplot(data['AGE'], kde=True)
# plt.ylabel('AGE')
# plt.show()

# 双变量
# sns.jointplot(data.AGE, data.RPM, kind='hex')
# plt.show()


# 多变量
# mul = data.loc[:, ['RPM','SALARY_MILLIONS','AGE','POINTS']]
# sns.pairplot(mul)
# plt.show()

def cut_age(df):
    if df.AGE <= 24:
        return 'young'
    elif df.AGE >= 30:
        return 'old'
    else:
        return 'best'


data['cut_age'] = data.apply(lambda x: cut_age(x), axis=1)
# print(data)

# plt.figure(figsize=(10, 8))
# x1 = data.loc[data['cut_age'] == 'young'].SALARY_MILLIONS
# y1 = data.loc[data['cut_age'] == 'young'].RPM
# plt.plot(x1, y1, '^')
#
# x2 = data.loc[data.cut_age == "best"].SALARY_MILLIONS
# y2 = data.loc[data.cut_age == "best"].RPM
# plt.plot(x2, y2, "^")
#
# x3 = data.loc[data.cut_age == "young"].SALARY_MILLIONS
# y3 = data.loc[data.cut_age == "young"].RPM
# plt.plot(x3, y3, ".")
# plt.show()

group1 = data.groupby(by='cut_age').agg({'SALARY_MILLIONS': np.max})
group2 = data.groupby(by='cut_age').agg({'SALARY_MILLIONS': np.mean})
# print(group1)

group3 = data.groupby(by='TEAM').agg({'SALARY_MILLIONS': np.mean})
# print(group3)
group3 = group3.sort_values(by='SALARY_MILLIONS', ascending=False)
# print(group3)

# 按照分球队分年龄段，上榜球员降序排列，如上榜球员数相同，则按效率值降序排列。
group4 = data.groupby(by=['TEAM', 'cut_age']).agg({'SALARY_MILLIONS': np.mean, 'PLAYER': np.size, 'RPM': np.mean})
# print(group4)
group4 = group4.sort_values(by=['PLAYER', 'RPM'], ascending=False)
# print(group4)

# 按照球队综合实力排名
group5 = data.groupby(by='TEAM').agg({'SALARY_MILLIONS': np.mean,
                                      'RPM': np.mean,
                                      'PLAYER': np.size,
                                      'POINTS': np.mean,
                                      'eFG%': np.mean,
                                      'MPG': np.mean,
                                      'AGE': np.mean})
# print(group5)
group5 = group5.sort_values(by='RPM', ascending=False)
# print(group5)

isin = data.TEAM.isin(['GS', 'CLE', 'SA', 'LAC', 'OKC', 'UTAH', 'CHA', 'TOR', 'NO', 'BOS'])
# print(isin)
# print(data['TEAM'].unique())

# plt.figure(figsize=(20, 10))
# sns.set_style('whitegrid')
#
data_part = data[isin]
# # print(data_part)
# plt.subplot(3, 1, 1)
# sns.boxplot(x='TEAM', y='SALARY_MILLIONS', data=data_part)
#
# plt.subplot(3, 1, 2)
# sns.boxplot(x='TEAM', y='AGE', data=data_part)
#
# plt.subplot(3, 1, 3)
# sns.boxplot(x='TEAM', y='MPG', data=data_part)
# plt.show()

plt.figure(figsize=(20, 10))
# print(data.columns)
sns.set_style('whitegrid')
plt.subplot(3, 1, 1)
sns.violinplot(x='TEAM', y='3P%', data=data_part)

plt.subplot(3, 1, 2)
sns.violinplot(x='TEAM', y='eFG%',data=data_part)

plt.subplot(3, 1, 3)
sns.violinplot(x="TEAM", y="POINTS", data=data_part)
plt.show()