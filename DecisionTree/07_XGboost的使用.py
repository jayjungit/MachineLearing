import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import xgboost as xgb
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import  warnings

warnings.filterwarnings('ignore')

# 使用方法一
# X, y = datasets.load_wine(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2)
# model = XGBClassifier(learning_rate=0.1,  # 学习率，控制每次迭代更新权重时的步长，默认0.3。值越小，训练越慢。
#                       use_label_encoder=False,
#                       n_estimators=10,  # 总共迭代的次数，即决策树的个数
#                       max_depth=5,  # 深度
#                       min_child_weight=1,  # 默认值为1,。值越大，越容易欠拟合；值越小，越容易过拟合
#                       gamma=0,  # 惩罚项系数，指定节点分裂所需的最小损失函数下降值。
#                       subsample=0.8,  # 训练每棵树时，使用的数据占全部训练集的比例。默认值为1，典型值为0.5-1。防止overfitting。
#                       colsample_bytree=0.8,
#                       objective='binary:logistic',  # 目标函数
#                       eval_metric=['merror'],  # 验证数据集评判标准
#                       nthread=4, )  # 并行线程数
#
# eval_set = [(X_test, y_test), (X_train, y_train)]
# model.fit(X_train, y_train, eval_set=eval_set, verbose=True)
# print('XGboost得分：',model.score(X_test, y_test))


# 使用方法二
# params = {'leaning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 5,
#          'min_child_weight': 1, 'gamma': 0, 'subsample': 0.8, 'colsample_bytree': 0.8, 'verbosity': 0,
#          'objective': 'multi:softprob'}
#
# model = XGBClassifier(**params)
# model.fit(X_train, y_train, early_stopping_rounds=20, eval_metric='merror', eval_set=[(X_test, y_test)])
# print("XGboost的得分：", model.score(X_test, y_test))


# 使用方法三
# DMatrix是XGBoost中使用的数据矩阵。DMatrix是XGBoost使用的内部数据结构，它针对内存效率和训练速度进行了优化
# dtrain = xgb.DMatrix(data=X_train, label=y_train)
# dtest = xgb.DMatrix(data=X_test, label=y_test)
#
# # 指定参数
# param = {'learning_rate': 0.1, 'use_label_encoder': False, 'max_depth': 5,
#          'min_child_weight': 1, 'gamma': 0, 'subsample': 0.8, 'eval_metric': ['merror', 'mlogloss'],
#          'colsample_bytree': 0.1, 'verbosity': 0, 'objective': 'multi:softmax', 'num_class': 3}
#
# num_round = 20
# evals = [(dtrain, 'train'), (dtest, 'eval')]
# bst = xgb.train(param, dtrain, num_round, evals=evals)
# y_ = bst.predict(dtest)
# print('XGboost的准确率为', accuracy_score(y_test, y_))


# XGboost实战
train = pd.read_csv("../data/train_modified.csv")
test = pd.read_csv("../data/test_modified.csv")
train.drop(labels='ID', axis=1, inplace=True)
test.drop(labels='ID', axis=1, inplace=True)

label = 'Disbursed'
cols = [i for i in train.columns if i not in [label]]


# 构建训练函数
def model_fit(model, dtrain, dtest, cols, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = model.get_xgb_params()
        xgb_train = xgb.DMatrix(data=dtrain[cols].values, label=dtrain['Disbursed'].values)
        xgb_test = xgb.DMatrix(data=dtest[cols].values)
        # print('model.get_xgb_params():',model.get_xgb_params())
        # cvresult = xgb.cv(xgb_param, xgb_train, num_boost_round=model.get_xgb_params()['n_estimators'],
        #                   nfold=cv_folds, early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
        cvresult = xgb.cv(xgb_param, xgb_train, num_boost_round=100,
                          nfold=cv_folds, early_stopping_rounds=early_stopping_rounds, verbose_eval=False)

        model.set_params(n_estimators=cvresult.shape[0])
    model.fit(dtrain[cols], dtrain['Disbursed'])
    y_ = model.predict(dtrain[cols])
    proba_ = model.predict_proba(dtrain[cols])[:, 1]
    print(proba_)
    print('___________________')
    print("该模型的表现：")
    print('准确率（训练集）：%.4g' % metrics.accuracy_score(dtrain[label], y_))
    print('AUC(训练集)得分：%f' % metrics.roc_auc_score(dtrain[label], proba_))

    # 可视化特征的重要性
    # feature_img = pd.Series(model.get_booster().get_fscore()).sort_values(ascending=False)
    # feature_img.plot(kind='bar', title='Feature Importance')
    # plt.ylabel('Feature Importance Score')
    # plt.show()


xgb1 = XGBClassifier(learning_rate=0.1,
                     use_label_encoder=False,
                     n_estimators=1000,
                     max_depth=5,
                     min_child_weight=1,
                     gamma=0,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     objective='binary:logistic',
                     nthread=4,
                     scale_pos_weight=1,
                     reg_alpha=0,
                     eval_metric=['error', 'auc'],
                     verbosity=0)
# model_fit(xgb1, train, test, cols)

# 筛选最优参数一
# param_grid = {"max_depth": range(2, 10, 2), 'min_child_weight': range(1, 6, 2)}
# model = XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=5, use_label_encoder=False,
#                       min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
#                       objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27,
#                       verbosity=0)
# gsearch1 = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', n_jobs=-1, cv=5)
# gsearch1.fit(train[cols], train[label])
# print('本次最佳参数:', gsearch1.best_params_)
# print('本次最优得分为:', gsearch1.best_score_)


# 筛选参数二
# param_grid = {'gamma': [i / 10.0 for i in range(0, 5)]}
# model = XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=5, use_label_encoder=False,
#                       min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
#                       objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27,
#                       verbosity=0)
# gsearch2 = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5, scoring='roc_auc')
# gsearch2.fit(train[cols], train[label])
# print('本次最优参数为：', gsearch2.best_params_)
# print('本次最优得分为：', gsearch2.best_score_)


# 筛选参数三
param_grid = {'subsample': [i/10 for i in range(0, 10)], 'closample_bytree': [i/10 for i in range(0, 10)]}
model = XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=5, use_label_encoder=False,
                      min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                      objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27,
                      verbosity=0)

gsearch3 = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5, scoring='roc_auc')
gsearch3.fit(train[cols], train[label])
print('本次最优的参数为：', gsearch3.best_params_)
print('本次最优的分数为：', gsearch3.best_score_)
