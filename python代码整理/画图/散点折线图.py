import pandas as pd
import matplotlib.pyplot as plt

#设置字体
plt.rcParams['font.sans-serif'] = ['PingFang HK']
plt.rcParams['axes.unicode_minus'] = False

#读取数据
y_pre=pd.read_excel('/Users/faguangnanhai/Desktop/预测路线成本.xlsx')#读取预测值数据
y_true=pd.read_excel('/Users/faguangnanhai/Desktop/真实路线成本.xlsx')#读取真实值数据
x_labol=pd.read_excel('/Users/faguangnanhai/Desktop/x.xlsx')#读一个x轴数据
##设置折线图与散点图
plt.plot(x_labol,y_true,linewidth=0.5,color='orange')
plt.scatter(x_labol,y_pre,s=6,color='deepskyblue')
##设置x，y坐标以及标注
plt.xlabel("样本数据",fontsize=20)
plt.ylabel("预测值与真实值价格",fontsize=20)
plt.legend(["真实值","预测值"])
plt.show()