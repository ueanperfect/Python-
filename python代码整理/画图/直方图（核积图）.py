import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.font_manager import FontProperties
##设置字体
myfont=FontProperties(fname=r'/Users/faguangnanhai/Desktop/输入文件文件夹/字体/PingFang.ttc',size=14)
sns.set(font=myfont.get_name())
plt.rcParams['font.sans-serif'] = ['PingFang HK']
plt.rcParams['axes.unicode_minus'] = False

##读取数据
data=pd.read_excel("/Users/faguangnanhai/Desktop/输出文件文件夹/得分.xlsx")
sns.utils.axlabel('指导价格得分大小', '指导价格得分分布')#设置x，y轴
sns.distplot(data,color='red')
plt.show()

