from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

##读取数据（在这里将想要读取的数据带入）
all_data = pd.read_excel("")
##选择自变量的位置
x=all_data.iloc[:,:5]
y=all_data.iloc[:,5]

##处理数据（标准化数据）
scale_x = StandardScaler()
x1=scale_x.fit_transform(x)
scale_y = StandardScaler()
y=np.array(y).reshape(-1,1)
y1=scale_y.fit_transform(y)
y1=y1.ravel()

x_train1,x_test1,y_train1,y_test1 = train_test_split(x1,y1,test_size = 0.25,random_state = 11)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 11)

##平滑化数据
y_log=np.log(y)
x_train,x_test,y_train_log,y_test_log = train_test_split(x,y_log,test_size = 0.25,random_state = 11)

##建立模型
models = [LinearRegression(), KNeighborsRegressor(), SVR(), Ridge(), Lasso(), MLPRegressor(alpha=20),
          DecisionTreeRegressor(), ExtraTreeRegressor(), XGBRegressor(), RandomForestRegressor(), AdaBoostRegressor(),
          GradientBoostingRegressor(), BaggingRegressor()]
models_str = ['LinearRegression', 'KNNRegressor', 'SVR', 'Ridge', 'Lasso', 'MLPRegressor', 'DecisionTree', 'ExtraTree',
              'XGBoost', 'RandomForest', 'AdaBoost', 'GradientBoost', 'Bagging']
score_adapt = []

##开始计算
for name, model in zip(models_str, models):
    if name in ['LinearRegression', 'Ridge', 'ExtraTree']:
        print('开始训练模型：' + name + ' 平滑处理')
        model = model
        model.fit(x_train, y_train_log)
        y_pred = model.predict(x_test)
        score = model.score(x_test, y_test_log)
        score_adapt.append(str(score)[:5])
        print(name + ' 得分:' + str(score))

    elif name in ['SVR', 'MLPRegressor', 'Bagging', 'AdaBoost', 'KNNRegressor']:
        print('开始训练模型：' + name + ' 标准化处理')
        model = model
        model.fit(x_train1, y_train1)
        y_pred = model.predict(x_test1)
        ypred_original = scale_y.inverse_transform(y_pred)
        score = model.score(x_test1, y_test1)
        score_adapt.append(str(score)[:5])
        print(name + ' 得分:' + str(score))

    else:
        print('开始训练模型：' + name + ' 普通')
        model = model
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        ypred_original = scale_y.inverse_transform(y_pred)
        score = model.score(x_test, y_test)
        score_adapt.append(str(score)[:5])
        print(name + ' 得分:' + str(score))
