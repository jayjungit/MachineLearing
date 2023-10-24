import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import SGDRegressor, LinearRegression


X = np.linspace(-1, 11, num=100)
y = (X - 5)**2 + 3*X - 12 + np.random.randn(100)
X = X.reshape(-1, 1)
plt.scatter(X, y)

X_test = np.linspace(-2, 12, num=200).reshape(-1, 1)


poly = PolynomialFeatures()

poly.fit(X, y)
X = poly.transform(X)
s = StandardScaler()
X = s.fit_transform(X)

model = SGDRegressor(penalty='l2', eta0=0.01)
model.fit(X, y)

X_test = poly.transform(X_test)
X_test_norm = s.transform(X_test)
y_test = model.predict(X_test_norm)

plt.plot(X_test[:, 1], y_test, color='green')
plt.show()