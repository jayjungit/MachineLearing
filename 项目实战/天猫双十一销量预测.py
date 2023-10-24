import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

plt.figure(figsize=(9, 6))
plt.rcParams['font.size'] = 10

X = np.arange(2009, 2020)
y = np.array([0.5, 9.36, 52, 191, 350, 571, 912, 1207, 1682, 2135, 2684])

x = X - X.mean()
print(x)
x = x.reshape(-1, 1)
poly = PolynomialFeatures(degree=3)
'''
    degree（默认为2）：用于指定生成多项式特征的次数。例如，如果 degree 为2，那么将生成一次项、二次项以及交叉二次项。

    interaction_only（默认为False）：如果设置为True，仅生成特征交叉项，不生成每个特征的幂项。这在希望仅考虑特征交互而不考虑特征自身幂的情况下很有用。

    include_bias（默认为True）：是否在生成的特征中包含常数项。如果设置为True，将生成全为1的特征列。

    order：控制生成的多项式特征的排序方式。默认为'C'，表示按照幂次降序排列，例如，x^2, x^1, x^0。如果设置为'F'，则按照幂次升序排列，例如，x^0, x^1, x^2。
'''

x = poly.fit_transform(x)
s = StandardScaler()
x_norm = s.fit_transform(x)

model = SGDRegressor(penalty='l2', eta0=0.01, max_iter=5000)
'''
    loss：指定损失函数的类型，例如均方误差（'squared_loss'）或绝对损失（'huber'）。默认为 'squared_loss'。

    penalty：指定正则化项的类型，例如 L2 正则化（'l2'）或 L1 正则化（'l1'）。默认为 'l2'。
    
    alpha：正则化项的强度，控制正则化的程度。较大的值表示更强的正则化。默认为 0.0001。
    
    max_iter：迭代的最大次数，用于控制训练的停止条件。默认为 1000。
    
    tol：用于控制损失的收敛阈值。当损失变化小于此值时，训练将停止。默认为 1e-3。
    
    epsilon：仅在 loss='huber' 时有用，用于指定 Huber loss 的阈值。默认为 0.1。
    
    learning_rate：学习率的类型，可以是 'constant'（常数学习率）、'optimal'（使用理论上的最佳学习率）、'invscaling'（逐渐降低学习率）等。默认为 'invscaling'。
    
    eta0：学习率初始值。只在 learning_rate='constant' 时有用。默认为 0.01。
    
    power_t：仅在 learning_rate='invscaling' 时有用，用于控制学习率逐渐降低的速度。默认为 0.25。
    
    shuffle：是否在每次迭代前打乱训练数据。默认为 True。
    
    random_state：用于设置随机数生成的种子，以确保结果的可重复性。
    
    verbose：控制训练过程中的详细程度，可以是 0（无输出）、1（偶尔输出）、2（详细输出）。默认为 0。
    
    warm_start：如果设置为 True，则在前一次训练的基础上继续训练。默认为 False。
'''
model.fit(x_norm, y)

x_test = np.linspace(-5, 8, 14).reshape(-1, 1)
x_test = poly.transform(x_test)
x_test_norm = s.transform(x_test)
y_test = model.predict(x_test_norm)

print(len(y_test))
plt.plot(x_test[:, 1], y_test, color='green')
# # plt.bar(x_test[:, 1], y_test, color='red')
plt.bar(x[:, 1], y)
plt.bar([6, 7, 8], y_test[-3:], color='red')
plt.text(8, y_test[-1] + 100, round(y_test[-1], 1), ha='center')
_ = plt.xticks(np.arange(-5, 9), np.arange(2009, 2023))
plt.ylim(0, 5000)
plt.show()
