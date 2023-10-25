import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor, ElasticNet, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split


data = pd.read_excel('../data/中国人寿.xlsx')

print(data.shape)
sns.kdeplot(data['charges'], shade=True, hue=data['sex'])
'''
    data：指定要绘制 KDE 图的数据，可以是 Series、DataFrame 或数组。

    x 和 y：分别表示数据的 x 和 y 轴。可以分别指定数据的两个维度，用于绘制二维 KDE 图。

    shade：控制是否为 KDE 图着色。默认为 True，表示用颜色填充 KDE 曲线下的区域。

    vertical：如果为 True，将生成垂直方向的 KDE 图，反之生成水平方向的图。默认为 False。

    bw_method：用于控制 KDE 估计带宽的方法，可以是字符串（例如，'scott'、'silverman'）或标量。较大的值表示较平滑的估计，较小的值表示更详细的估计。

    kernel：指定核函数的类型，例如 'gau'（高斯核函数，默认）、'cos'（余弦核函数）、'biw'（二元核函数）等。

    cumulative：如果为 True，将生成累积分布函数图。默认为 False。

    common_norm：控制 KDE 曲线的归一化方式。如果为 True，则所有的曲线共享一个归一化尺度。默认为 False。

    color：指定曲线的颜色。可以是颜色名称、十六进制颜色码或颜色缩写。

    label：指定曲线的标签，用于图例。

    legend：如果为 True，将显示图例。默认为 True。
'''
sns.kdeplot(data['charges'], shade=True, hue=data['region'])
sns.kdeplot(data['charges'], shade=True, hue=data['smoker'])
sns.kdeplot(data['charges'], shade=True, hue=data['children'], palette='Set1')

data = data.drop(['region', 'sex'], axis=1)


def convert(df, bmi):
    df['bmi'] = 'fat' if df['bmi'] > bmi else 'standard'
    return df


data = data.apply(convert, axis=1, args=(30, ))
data = pd.get_dummies(data)
x = data.drop(['charges'], axis=1)
y = data['charges']

# 特征升维
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
poly = PolynomialFeatures(degree=2, include_bias=False)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.fit_transform(x_test)

model_1 = LinearRegression()
model_1.fit(x_train_poly, y_train)
print('普通线性回归训练得分：', model_1.score(x_train_poly, y_train))
print('普通线性回归测试得分：', model_1.score(x_test_poly, y_test))
print('普通线性回归训练均方误差：', np.sqrt(mean_squared_error(y_train, model_1.predict(x_train_poly))))
print('普通线性回归测试均方误差：', np.sqrt(mean_squared_error(y_test, model_1.predict(x_test_poly))))

print('普通线性回归训练对数均方误差：', np.sqrt(mean_squared_log_error(y_test, model_1.predict(x_test_poly))))
print('普通线性回归测试对数均方误差：', np.sqrt(mean_squared_log_error(y_test, model_1.predict(x_test_poly))))


# 弹性网络回归(融合l1, l2)
model_2 = ElasticNet(alpha=0.2, l1_ratio=1, max_iter=50000)
'''
    alpha：控制正则化的强度，是 L1 和 L2 正则化的权衡因子。较小的 alpha 值表示更弱的正则化，较大的值表示更强的正则化。通常需要进行超参数调整以找到最佳的 alpha 值。

    l1_ratio：控制 L1 正则化的比例。如果 l1_ratio 为1，表示只使用 L1 正则化（Lasso），如果 l1_ratio 为0，表示只使用 L2 正则化（Ridge）。在这两者之间的值表示 L1 和 L2 正则化的组合。通常需要进行超参数调整以找到最佳的 l1_ratio 值。
    
    fit_intercept：表示是否拟合截距项（即线性模型中的截距项）。默认为 True。
    
    max_iter：迭代的最大次数。默认为 1000。
    
    tol：用于控制损失的收敛阈值。当损失变化小于此值时，训练将停止。默认为 1e-4。    
'''
model_2.fit(x_train_poly, y_train)
print('普 弹性网络训练得分：', model_2.score(x_train_poly, y_train))
print(' 弹性网络测试得分：', model_2.score(x_test_poly, y_test))
print(' 弹性网络训练均方误差：', np.sqrt(mean_squared_error(y_train, model_2.predict(x_train_poly))))
print(' 弹性网络测试均方误差：', np.sqrt(mean_squared_error(y_test, model_2.predict(x_test_poly))))

print(' 弹性网络训练对数均方误差：', np.sqrt(mean_squared_log_error(y_test, model_2.predict(x_test_poly))))
print(' 弹性网络测试对数均方误差：', np.sqrt(mean_squared_log_error(y_test, model_2.predict(x_test_poly))))






