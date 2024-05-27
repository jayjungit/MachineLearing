import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'KaiTi'

data = pd.read_csv('../data/链家北京租房数据.csv')
# print(data.head())
# print(data.shape)
# print(data.describe())
# print(data.info())
# 删除重复的值
data = data.drop_duplicates()
# print(data.shape)
data = data.dropna()
# print(data.shape)
# print(data['面积(㎡)'].values[0][:-2])
# values_list = []
# m_values = data['面积(㎡)'].values
# for i in m_values:
#     values_list.append(i[:-2])
#
# values_list = list(map(np.float32, values_list))
# data['面积(㎡)'] = values_list

data['面积(㎡)'] = data['面积(㎡)'].str[:-2].astype(np.float64)

house_data = data['户型'].values
temp_list = []
for i in house_data:
    temp = i.replace('房间', '室')
    temp_list.append(temp)

data['户型'] = temp_list
# print(data.tail())
# print(data['户型'].unique())


# 房源数量、位置分布分析
new_df = pd.DataFrame({"区域": data["区域"].unique(), "数量": [0] * 13})

group1 = data.groupby(by='区域').count()
new_df['数量'] = group1.values
# print(new_df)
# # 升序排序
group2 = new_df.sort_values(by='数量', ascending=False)
# print(group2)

house_data = data['户型']


def all_house(arr):
    key = data['户型'].unique()
    result = {}
    for k in key:
        mask = arr == k
        new_arr = arr[mask]
        v = new_arr.size
        result[k] = v
    return result


house = all_house(house_data)
house_data = dict((key, value) for key, value in house.items() if value > 50)
# print(house_data)

house_data = pd.DataFrame({'户型': [x for x in house_data.keys()], '数量': [x for x in house_data.values()]})
# print(house_data)

# plt.figure(figsize=(10, 6))
# plt.bar(house_data['户型'], house_data['数量'])
# # plt.yticks(range(11), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
# # plt.xlim(0, 2500)
# plt.ylabel('户型')
# plt.xlabel('数量')
# plt.title('北京市各区域租房数量统计')
# for x, y in enumerate(house_data['数量']):
#     plt.text(x-0.2, y+0.5, "%s"%y)
# plt.show()


# 平均租金分析
dff_all = pd.DataFrame({'区域': data['区域'].unique(), '房租总金额': [0] * 13, '总面积': [0] * 13})
# print(dff_all)
# print(data.head())

sum_price = data['价格(元/月)'].groupby(by=data['区域']).sum()

sum_area = data['面积(㎡)'].groupby(by=data['区域']).sum()
# print(sum_area)

dff_all['房租总金额'] = sum_price.values
dff_all['总面积'] = sum_area.values
# print(dff_all)

# 计算各个区域每平方米的房租
dff_all['每平方米的租金'] = np.round(dff_all['房租总金额'] / dff_all['总面积'], 2)
# print(dff_all)
df_merge = pd.merge(group2, dff_all, on=['区域'])
# print(df_merge)


# fig = plt.figure(figsize=(10, 6), dpi=100)
# ax1 = fig.add_subplot(111)
# ax1.plot(df_merge['区域'], df_merge['每平方米的租金'], 'or-', label='价格', )
# for x, y in enumerate(df_merge['每平方米的租金']):
#     ax1.text(x, y+1, '%s'%y)
# ax1.set_ylabel("价格")
# ax1.legend('upper_right')
#
# ax2 = ax1.twinx()
# ax2.bar(df_merge['区域'], df_merge['数量'], color='green')
# ax2.set_ylabel('数量')
# ax2.legend("upper_left")

# num = df_merge["数量"]
# price = df_merge["每平方米的租金"]
# lx = df_merge["区域"]
# l = [i for i in range(13)]
#
# fig = plt.figure(figsize=(10, 8), dpi=100)
#
# # 显示折线图
# ax1 = fig.add_subplot(111)
# ax1.plot(l, price, "or-", label="价格")
# for i, (_x, _y) in enumerate(zip(l, price)):
#     ax1.text(_x+0.2, _y, f'{price[i]}', fontsize=9, color='red')
# ax1.set_ylim([0, 160])
# ax1.set_ylabel("价格")
# ax1.legend(loc="upper right")
#
# # 显示条形图
# ax2 = ax1.twinx()
# ax2.bar(l, num, label="数量", alpha=0.2, color="green")
# ax2.set_ylabel("数量")
# ax2.legend(loc="upper left")
# plt.xticks(l, lx)
#
# plt.show()


# 面积划分
area_divide = [1, 30, 50, 70, 90, 120, 140, 160, 1200]
area_cut = pd.cut(list(data["面积(㎡)"]), area_divide)
area_cut = area_cut.describe()
area_per = (area_cut['freqs'].values) * 100
# print(area_per)

labels = ['30平米以下', '30-50平米', '50-70平米', '70-90平米',
          '90-120平米', '120-140平米', '140-160平米', '160平米以上']

plt.figure(figsize=(10, 6), dpi=100)
plt.pie(area_per, labels=labels, autopct='%.2f %%')
plt.legend(loc='upper right')
plt.show()
