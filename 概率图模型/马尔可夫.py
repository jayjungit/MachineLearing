# HMM模型（隐马尔可夫模型）
'''
下面抽取5次，得到已知的观察序列，
来对HMM的参数进行估计，即使用MultinomialHMM进行参数的训练
'''
# import numpy as np
# import hmmlearn.hmm as hmm
#
# states = ['盒子1', '盒子2', '盒子3']
# obs = ['红球', '白球']  # 0表示红球，1表示白球
# n_states = len(states)
# m_obs = len(obs)
#
# # 将二元观察结果转换为多项分布的格式
# X2_multinomial = np.array([
#     [1, 1, 0, 0, 0],  # 盒子1: 红球、白球
#     [1, 0, 0, 0, 1],  # 盒子2: 红球、白球
#     [0, 0, 1, 1, 0],  # 盒子3: 红球、白球
#     [1, 0, 0, 1, 0],  # 盒子1: 红球、白球
#     [1, 0, 0, 0, 1]   # 盒子2: 红球、白球
# ])
#
# model = hmm.MultinomialHMM(n_components=n_states, n_iter=100000, tol=0.001, algorithm='viterbi')
# model.fit(X2_multinomial)
# print("输出根据数据训练出来的π")
# print(model.startprob_.round(3))
# print("输出根据数据训练出来的A")
# print(model.transmat_.round(3))
# print("输出根据数据训练出来的B")
# print(model.emissionprob_.round(3))


# import numpy as np
# import hmmlearn.hmm as hmm
#
# # 首先定义变量
# status = ['盒子1', '盒子2', '盒子3']   # 隐藏的状态集合
# obs = ['红球', '白球']     # 观察值集合
# n_status = len(status)    # 隐藏状态的长度
# m_obs = len(obs)          # 观察值的长度
#
# # 初始概率分布： π 表示初次抽时，抽到1盒子的概率是0.2，抽到2盒子的概率是0.4，抽到3盒子的概率是0.4
# start_probability = np.array([0.2, 0.4, 0.4])
#
# # 状态转移概率矩阵 A[0][0]=0.5 表示当前我抽到1盒子，下次还抽到1盒子的概率是0.5
# transition_probability = np.array([[0.5, 0.2, 0.3],
#                                    [0.3, 0.5, 0.2],
#                                    [0.2, 0.3, 0.5]])
#
# # 观测概率矩阵 B：B[2][0]=0.7,表示第三个盒子抽到红球概率0.7，B[2][1]=0.3，表示第三个盒子抽到白球概率0.3
# emission_probalitity = np.array([[0.5, 0.5],
#                                  [0.4, 0.6],
#                                  [0.7, 0.3]])
#
# # 下面开始定义模型
# '''
# hmmlearn中主要有两种模型，分布为：GaussianHMM和MultinomialHMM；
# 如果观测值是连续的，那么建议使用GaussianHMM，否则使用MultinomialHMM
#
# 参数：
# 初始的隐藏状态概率π参数为: startprob；
# 状态转移矩阵A参数为: transmat;
# 状态和观测值之间的转移矩阵B参数为:
# emissionprob_(MultinomialHMM模型中)或者在GaussianHMM模型中直接给定均值(means)和方差/协方差矩阵(covars)
#
# '''
# # 观测值，球是红或者黑，是离散的，n_status隐藏状态的长度
# model = hmm.MultinomialHMM(n_components=n_status)
# model.startprob_ = start_probability
# model.transmat_ = transition_probability
# model.emissionprob_ = emission_probalitity
#
# '''
# 下面运行viterbi预测问题。已知观察序列（红球 白球 红球），
# 求什么样的隐藏状态序列（盒子2 盒子3 盒子2）最可能生成一个给定的观察序列。
#
# status = ['盒子1', '盒子2', '盒子3']
# obs = ['红球', '白球']
# '''
# se = np.array([[0, 1, 0]]).T   # （红球 白球 红球）
# logprob, box_index = model.decode(se, algorithm='viterbi')
# print("颜色:", end="")
# print(" ".join(map(lambda t: obs[t], [0, 1, 0])))
# print("盒子:", end="")
# print(" ".join(map(lambda t: status[t], box_index)))
# print("概率值:", end="")
# print(np.exp(logprob)) # 这个是因为在hmmlearn底层将概率进行了对数化，防止出现乘积为0的情况


# HMM实战
# import hmmlearn.hmm as hmm
# import numpy as np
# import pandas as pd
# from hmmlearn.hmm import GaussianHMM
#
# data = pd.read_csv('../data/apple_stock.csv')
# data['Date'] = pd.to_datetime(data['Date'], format='%/Y%/m%/d', errors='ignore')
#
# volume = data['Volume'].values[1:]
#
# close_v = data['Close'].values
# diff = np.diff(close_v)
# X = np.column_stack([diff, volume])
#
# model = GaussianHMM(n_components=3, covariance_type='diag', n_iter=100000)
# model.fit(X)
# print('每个隐含层均值和方差数据：')
# for i in range(model.n_components):
#     print('第{0}号隐藏状态'.format(i))
#     print('mean = ', model.means_[i])
#     print('var = ', np.diag(model.covars_[i]))
#     print('---------------')
#
# '''
# 每个隐含层均值和方差数据：
# 第0号隐藏状态
# mean =  [-1.14524842e-01  1.23845771e+08]
# var =  [3.84678157e+00 2.24454700e+15]
# ---------------
# 第1号隐藏状态
# mean =  [1.68138743e-01 2.79598061e+07]
# var =  [1.00638562e+00 5.12991123e+13]
# ---------------
# 第2号隐藏状态
# mean =  [5.04087051e-02 5.66296542e+07]
# var =  [2.81382699e+00 2.27530289e+14]
# ---------------
# '''
#
#
# # 状态转移矩阵
# print('Transition matrix')
# print(np.round(model.transmat_, 3))


# crf 命名体识别实战
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
import nltk


# nltk.download('conll2002', download_dir='../data')
# 查看数据的类别
# print(nltk.corpus.conll2002.fileids())
train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))


# 获取词的特征
def word2features(sents, i):
    word = sents[i][0]
    postag = sents[i][1]

    # 当前词的特征
    features = {
        'word.lower': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word_upper': word.upper(),
        'word_istitle': word.istitle(),
        'word_isdigit': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2]
    }

    # 前一个词的特征
    if i > 0:
        word1 = sents[i-1][0]
        postag1 = sents[i-1][1]
        features.update({
            '-1:word_lower': word1.lower(),
            '-1:world_istitle': word1.istitle(),
            '-1:word_isupper': word1.isupper(),
            '-1:postg1': postag1,
            '-1:postag[:2]': postag1[:2]
        })
    else:
        features['BOS'] = True

    # 后一个词的特征
    if i < len(sents) - 1:
        word1 = sents[i+1][0]
        postag1 = sents[i+1][1]

        features.update({
            '+1:word_lower': word1.lower(),
            '+1:world_istitle': word1.istitle(),
            '+1:word_isupper': word1.isupper(),
            '+1:postg1': postag1,
            '+1:postag[:2]': postag1[:2]
        })
    else:
        features['EOS'] = True
    return features


# 文本数据特征获取
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2label(sent):
    return [label for token, postag, label in sent]


x_train = [sent2features(s) for s in train_sents]
y_train = [sent2label(s) for s in train_sents]
x_test = [sent2features(s) for s in test_sents]
y_test = [sent2label(s) for s in test_sents]


model = CRF(c1='0.1', c2=0.1, max_iterations=100, all_possible_transitions=True)
model.fit(x_train, y_train)
print("准确率：", model.score(x_test, y_test))
y_ = model.predict(x_test)
print("句子为：", test_sents[:1])
print("------------------------------------")
print("预测值为：", y_[:1])
print("------------------------------------")
print("真实值为：", y_test[:1])
