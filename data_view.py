# 导入常用库
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据集australian-1
CreditCard = pd.read_csv('Data/australian-1.csv')
CreditCard.head()
# 缺失值检查
print(CreditCard.isnull().sum())
# 异常值检查
CreditCard.info()
# Class列检查
CreditCard.Class.unique()
# 查看Class列不平衡程度
BarNum = CreditCard.Class.value_counts()
print(BarNum)
BarNum.plot.bar()
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Sample Distribution')
plt.text(0, BarNum[0], BarNum[0], fontsize=12, horizontalalignment='center', verticalalignment='bottom')
plt.text(1, BarNum[1], BarNum[1], fontsize=12, horizontalalignment='center', verticalalignment='bottom')
plt.show()


# 读取数据集german-1
CreditCard = pd.read_csv('Data/german-1.csv')
CreditCard.head()
# 缺失值检查
print(CreditCard.isnull().sum())
# 异常值检查
CreditCard.info()
# Class列检查
CreditCard.Class.unique()
# 查看Class列不平衡程度
BarNum = CreditCard.Class.value_counts()
print(BarNum)
BarNum.plot.bar()
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Sample Distribution')
plt.text(1, BarNum[2], BarNum[2], fontsize=12, horizontalalignment='center', verticalalignment='bottom')
plt.text(2, BarNum[1], BarNum[1], fontsize=12, horizontalalignment='center', verticalalignment='bottom')
plt.show()

# 读取数据集 barudata-1.csv
CreditCard = pd.read_csv('Data/barudata-1.csv')
CreditCard.head()
# 缺失值检查
print(CreditCard.isnull().sum())
# 异常值检查
CreditCard.info()
# Class列检查
CreditCard.Class.unique()
# 查看Class列不平衡程度
BarNum = CreditCard.Class.value_counts()
print(BarNum)
BarNum.plot.bar()
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Sample Distribution')
plt.text(1, BarNum[0], BarNum[0], fontsize=12, horizontalalignment='center', verticalalignment='bottom')
plt.text(0, BarNum[1], BarNum[1], fontsize=12, horizontalalignment='center', verticalalignment='bottom')
plt.show()

# # 查看变量的概率密度分布
# plt.figure(figsize=(20, 20))
# for i in range(0, CreditCard.shape[1] - 1):
#     plt.subplot(6, 5, i + 1)
#     sns.displot(CreditCard.iloc[:, i], kde=True)
#     plt.show()
