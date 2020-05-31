import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
##读取字体
plt.rcParams['font.sans-serif'] = ['PingFang HK']
plt.rcParams['axes.unicode_minus'] = False

##读取相关性系数表
table = pd.read_excel("")#相关性系数表excel文件
fig, ax = plt.subplots(figsize = (10,10))

#二维的数组的热力图，横轴和数轴的ticklabels要加上去的话，既可以通过将array转换成有column
#和index的DataFrame直接绘图生成，也可以后续再加上去。后面加上去的话，更灵活，包括可设置labels大小方向等。
sns.heatmap(table.iloc[:,1:13], annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True, square=True, cmap="YlGnBu")

##设置x，y轴名称以及大小
ax.set_ylabel('变量顺序', fontsize = 20)
ax.set_title('皮尔逊相关性系数',fontsize=20)
plt.show()