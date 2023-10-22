import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


X = np.linspace(-1, 11, num=100)
y = (X - 3)**2 + 3 * X - 12 + np.random.randn(100)
X = X.reshape(-1, 1)
plt.scatter(X, y)


X_test = np.linspace(-2, 12, num=200).reshape(-1, 1)
model_1 = LinearRegression()
model_1.fit(X, y)
y_test1 = model_1.predict(X_test)
plt.plot(X_test, y_test1, color='red')

# 升维
X = np.concatenate([X, X**2], axis=1)
model_2 = LinearRegression()
model_2.fit(X, y)
X_test2 = np.concatenate([X_test, X_test**2], axis=1)
y_test2 = model_2.predict(X_test2)
plt.plot(X_test2[:, 0], y_test2, color='green')
plt.show()

