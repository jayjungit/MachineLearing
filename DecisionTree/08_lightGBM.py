import lightgbm as lgb
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error


data = datasets.load_iris()
X = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = lgb.LGBMRegressor(objective='regression', learning_rate=0.1, n_estimators=40)
model.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric='l1')

print(model.score(x_test, y_test))


model = lgb.LGBMRegressor(num_leaves=31)
params = {'learning_rate': [0.01, 0.1, 1],
          'n_estimators': [20, 20, 40]}

gsc = GridSearchCV(estimator=model, param_grid=params, cv=4)
gsc.fit(x_train, y_train)
print("最好的参数为：", gsc.best_params_)

