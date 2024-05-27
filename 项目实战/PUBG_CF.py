import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# train_data = pd.read_csv('../data/PUBG_data/train_V2.csv')
# train_data = train_data.drop(2744604)
# # print(data.shape)
#
# # print(data.head())
# # print(data.isnull().sum())
# # print(data.info())
# # print(data[data['winPlacePerc'].isnull()])
#
# # data = data.drop(2744604)
# # print(data.isnull().sum())
# # train_sample = data.sample(frac=0.1)
# # train_sample.to_csv('../data/PUBG_data/train_sample.csv', index=False)
#
# # train_data = pd.read_csv('../data/PUBG_data/train_sample.csv')
# # print(train_data.shape)
# # print(train_data.head())
# # print(len(train_data['matchId'].unique()))
# # print(train_data.columns)
#
# # 特征数据规范化处理
# # 查看每场比赛参加的人数
# count = train_data.groupby(by='matchId')['matchId'].transform("count")
# train_data['playersJoined'] = count
# # print(train_data.head())
#
# # print(train_data['playersJoined'].sort_values())
# # plt.figure(figsize=(20, 8))
# # sns.countplot(x=train_data['playersJoined'])
# # plt.xticks(fontsize=8, color='red', rotation=45)  # 设置字体大小为12，字体颜色为红色
# # plt.xticks(rotation=45)  # 设置字体旋转角度为45度
# # plt.grid()
# # plt.show()
#
# # 每场比赛人数少于75人的
# # sns.countplot(train_data[train_data['playersJoined'] >= 75]['playersJoined'])
# # plt.grid()
# # plt.show()
#
# # 规范化输出的数据
# train_data["killsNorm"] = train_data["kills"] * ((100-train_data["playersJoined"])/100+1)
# train_data['damageDealtNorn'] = train_data['damageDealt'] * (100 - train_data['playersJoined'] / 100 + 1)
# train_data['maxPlaceNorn'] = train_data['maxPlace'] * (100 - train_data['playersJoined'] / 100 + 1)
# train_data['matchDurationNorn'] = train_data['matchDuration'] * (100 - train_data['playersJoined'] / 100 + 1)
# # print(train_data.head())
#
# column_list = ['damageDealt', 'damageDealtNorn', 'maxPlace', 'maxPlaceNorn', 'matchDuration', 'matchDurationNorn']
# # print(train_data[column_list][:10])
#
# # 合并部分变量
# train_data['healsandboosts'] = train_data['heals'] + train_data['boosts']
# train_data['totalDistance'] = train_data['walkDistance'] + train_data['swimDistance'] + train_data['rideDistance']
# # print(train_data.head())
#
#
# # print((train_data['totalDistance'] == 0).sum())
# # print(train_data.shape)
#
# train_data['killwithoutMoving'] = (train_data['kills'] > 0) & (train_data['totalDistance'] == 0)
# # print(train_data[(train_data['killwithoutMoving'] == True)][:5])
# # train_data.drop(train_data['totalDistance'] == 0, axis=0)
# # print(train_data.shape)
#
# # 删除异常值
# # 删除有击杀木移动的数据
# # print(train_data[train_data['killwithoutMoving'] == True].index)
# train_data.drop(train_data[train_data['killwithoutMoving'] == True].index, inplace=True)
# # print(train_data.shape)
#
# # 删除驾车击杀数量大于10 的数据
# train_data.drop(train_data[train_data['roadKills'] > 10].index, inplace=True)
# # print(train_data.shape)
#
# # 删除玩家在一局中击杀数量大于30的数据
# train_data.drop(train_data[train_data['kills'] > 30].index, inplace=True)
# # print(train_data.shape)
#
# # 删除爆头率异常的数据
# train_data['headshot_rate'] = train_data['headshotKills'] / train_data['kills']  # 计算爆头率
# train_data['headshot_rate'].fillna(value=0, inplace=True)  # 用0填充空数据
# # print(train_data.head())
# train_data.drop(train_data[(train_data['headshot_rate'] == 1) & (train_data['kills'] > 9)].index, inplace=True)
# # print(train_data.shape)
#
# # 删除最远距离击杀的异常数据
# train_data.drop(train_data[train_data['longestKill'] >= 1000].index, inplace=True)
# # print(train_data.shape)
#
# # 删除移动异常的数据
# train_data.drop(train_data[train_data['walkDistance'] >= 10000].index, inplace=True)
# # print(train_data.shape)
#
# train_data.drop(train_data[train_data['rideDistance'] >= 20000].index, inplace=True)
# train_data.drop(train_data[train_data['swimDistance'] >= 2000].index, inplace=True)
# # print(train_data.shape)
#
# # 删除武器收集异常的数据
# train_data.drop(train_data[train_data['weaponsAcquired'] >= 80].index, inplace=True)
# # 删除使用治疗药品的异常数据
# train_data.drop(train_data[train_data['heals'] >= 80].index, inplace=True)
# # print(train_data.shape)
#
# # 类别型数据处理
# # 比赛的类型one-hot编码
# train_data = pd.get_dummies(train_data, columns=['matchType'])
# # print(train_data.head())
#
# # print(train_data.filter(regex='matchType'))
#
# # 对groupId, matchId数据库进行处理
# train_data['groupId'] = train_data['groupId'].astype('category')
# train_data['groupId_cat'] = train_data['groupId'].cat.codes
# # print(train_data)
#
# train_data['matchId'] = train_data['matchId'].astype('category')
# train_data['matchId_cat'] = train_data['matchId'].cat.codes
# # print(train_data.head())
# train_data.drop(columns=['groupId', 'matchId'], axis=1, inplace=True)
# # print(train_data.shape)
#
#
# # 取部分数据来用
# df_sample = train_data.sample(100000)
# df_sample.to_csv('../data/PUBG_data/train_sample.csv', index=False)


df_sample = pd.read_csv('../data/PUBG_data/train_sample.csv')
# print(df_sample.shape)
# print(df_sample.head())
# print(df_sample.info())
y = df_sample['winPlacePerc'].astype(np.float64)
X = df_sample.drop(columns=['winPlacePerc', 'Id'], axis=1)
# print(X.shape, y.shape)

# 用随机森林进行建模
# x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
# model1 = RandomForestRegressor(n_estimators=100,
#                                min_samples_leaf=3,
#                                n_jobs=-1,
#                                max_features='sqrt')
# model1.fit(x_train, y_train)
# y_ = model1.predict(x_valid)
# print(model1.score(x_valid, y_valid))
# print(mean_absolute_error(y_valid, y_))
# # print(model1.feature_importances_)
# df = pd.DataFrame({'columns': x_train.columns, 'param': model1.feature_importances_})
# df = df.sort_values('param', ascending=False)
# # print(df)
# # df[:20].plot('columns', 'param', figsize=(20, 8), kind='barh')
# # plt.show()
#
# # 只取重要的特征
# df = df[df['param'] >= 0.005]
# to_keep = df['columns'].values
# # print(to_keep.size)
# X = df_sample[to_keep]
# x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
#
# model2 = RandomForestRegressor(n_estimators=100,
#                                min_samples_leaf=3,
#                                n_jobs=-1,
#                                max_features='sqrt')
# model2.fit(x_train, y_train)
# y_ = model2.predict(x_valid)
# print(model2.score(x_valid, y_valid))
# print(mean_absolute_error(y_valid, y_))


# 用lightGBM进行建模
x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
# model3 = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.5, n_estimators=20)
# model3.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], eval_metric='l1')
# y3 = model3.predict(x_valid, num_iteration=model3.best_iteration_)
# print("lightGBM的得分为：", model3.score(x_valid, y_valid))
# print('lightGBM的误差为：', mean_absolute_error(y_valid, y3))
# print('------------------------------------------------------')

# 模型二次调优
# model4 = lgb.LGBMRegressor(num_leaves=31)
# params_grid = {'learning_rate': [0.01, 0.1, 1],
#                'n_estimators': [40, 60, 80, 100, 200]}
#
# gsc = GridSearchCV(estimator=model4, param_grid=params_grid, cv=5, n_jobs=-1)
# gsc.fit(x_train, y_train)
# y4 = gsc.predict(x_valid)
# print("lightGBM的得分为：", gsc.score(x_valid, y_valid))
# print('lightGBM的误差为：', mean_absolute_error(y_valid, y4))
# print('最好的参数为：', gsc.best_params_)
# print('------------------------------------------------------')

# 三次调优
score = []
n_estimator = [100, 200, 300, 500]
for i in n_estimator:
    model5 = lgb.LGBMRegressor(n_estimators=i,
                               boosting_type='gbdt',
                               num_leaves=31,
                               learning_rate=0.1,
                               min_child_samples=20,
                               n_jobs=-1,
                               max_depth=5
                               )
    model5.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], eval_metric='l1')
    y5 = model5.predict(x_valid)
    score.append(model5.score(x_valid, y_valid))
    print('本次的mes为：', mean_absolute_error(y_valid, y5))

print(score)

plt.figure(figsize=(10, 6))
plt.plot(n_estimator, score, 'o-')
for i, (x, y) in enumerate(zip(n_estimator, score)):
    plt.text(x, y, '%.2f'%y)
plt.show()






