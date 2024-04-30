import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


train_data = pd.read_csv('../data/zhengqi_train.txt', sep='\t')
test_data = pd.read_csv('../data/zhengqi_test.txt', sep='\t')

X_train = train_data.iloc[:, :-1]
y = train_data['target']

X_test = test_data
model = GradientBoostingRegressor()
model.fit(X_train, y)
y_ = model.predict(X_test)
np.savetxt('result.txt', y_)