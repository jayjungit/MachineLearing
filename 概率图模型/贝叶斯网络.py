from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split

# X, y = datasets.load_iris(return_X_y=True)
# score = 0
#
# # 高斯分布朴素贝叶斯
# model = GaussianNB()
# for i in range(1000):
#     x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#     model.fit(x_train, y_train)
#     score += model.score(x_test, y_test) / 100
# print('高斯分布的得分：', score)
#
#
# # 伯努利高斯分布
# model = BernoulliNB()
# score = 0
# for i in range(100):
#     x_train, x_test, y_train, y_test = train_test_split(X, y)
#     model.fit(x_test, y_test)
#     score += model.score(x_test, y_test) / 100
# print('伯努利的得分:', score)
#
#
# # 多项式贝叶斯
# model = MultinomialNB()
# score = 0
# for i in range(100):
#     x_train, x_test, y_train, y_test = train_test_split(X, y)
#     model.fit(x_train, y_train)
#     score += model.score(x_test, y_test)/100
# print('多项式的得分：', score)


# # 词频-逆向文件频率(tf_idf)代码实现 ----> 用于文本任务的特征工程
# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# import numpy as np
#
#
# data = np.array(['政治 历史 地理 语文 政治', 'Python 计算机 语文 英语 数学'])
# cv = CountVectorizer()
# tf = TfidfTransformer()
# tf_idf = tf.fit_transform(cv.fit_transform(data))
# # 提取数据
# tf_idf_weight = tf_idf.toarray()
# print(tf_idf_weight)
# vocabulary = cv.vocabulary_
# print(sorted(vocabulary.items(), key=lambda x: x[1], reverse=False))


# 垃圾短信分类实战
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
import pandas as pd
import pickle
import bz2

# data = pd.read_csv('../data/messages.csv', sep='\t', header=None)
# data.rename({0: 'label', 1: 'messages'}, axis=1, inplace=True)
# cv = CountVectorizer()
# tf_idf = TfidfTransformer()
# X = cv.fit_transform(data['messages'])
# X = tf_idf.fit_transform(X).toarray()
# print(X.shape)
#
# y = data['label']
#
# x_train, x_test, y_train, y_test = train_test_split(X, y)
#
# model = GaussianNB()
# model.fit(x_train, y_train)
# print('高斯分布的得分：', model.score(x_test, y_test))
#
# model = BernoulliNB()
# model.fit(x_train, y_train)
# print('伯努利分布的分数：', model.score(x_test, y_test))


# 新闻分类
# news = datasets.fetch_20newsgroups(subset='all',remove=('headers', 'footers', 'quotes')) # 会自动下载
# 加载一部分数据
news = datasets.fetch_20newsgroups(data_home='../data', subset='all',
                                   remove=('headers', 'footers', 'quotes'),
                                   categories=['rec.motorcycles', 'rec.sport.hockey', 'talk.politics.guns'])

cv = CountVectorizer()
tf_idf = TfidfTransformer()

X = cv.fit_transform(news['data'])
X = tf_idf.fit_transform(X).toarray()
y = news['target']
x_train, x_test, y_train, y_test = train_test_split(X, y)

model1 = GaussianNB()
model1.fit(x_train, y_train)
print('高斯贝朴素叶斯分布的得分：', model1.score(x_test, y_test))

model2 = BernoulliNB()
model2.fit(x_train, y_train)
print('伯努利朴素贝叶斯分布的得分：', model2.score(x_test, y_test))
