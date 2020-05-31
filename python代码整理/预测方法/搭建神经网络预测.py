import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers,Sequential
from sklearn.preprocessing import StandardScaler

##读取文件
all_data = pd.read_excel("")#读取训练集样本
caculate_data=pd.read_excel("")##读取预测集样本
x=all_data.iloc[:,:113]
y=all_data.iloc[:,113]
caculate_data=caculate_data.iloc[:,:113]
TEST_SIZE=0.1
N=22

##数据标准化
y=np.array(y).reshape(-1,1)
scale_caculate=StandardScaler()
scale_x=StandardScaler()
scale_y=StandardScaler()
caculate_data=scale_caculate.fit_transform(caculate_data)
x1=scale_x.fit_transform(x)
y1=scale_y.fit_transform(y)

##创建训练以及数据集
x_train ,x_test,y_train,y_test=train_test_split(x1, y1, test_size=TEST_SIZE, random_state=N)

##将数据转化为tensorflow文件
x_caculate=tf.convert_to_tensor(caculate_data,dtype=tf.float32)
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
x_test = tf.cast(x_test, dtype=tf.float32)
y_test= tf.cast(y_test, dtype=tf.float32)

#通过from_tensor_slices将数据装载成dataset数据样本
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))

##样本数据集训练设置
train_db = train_db.shuffle(60000)
test_db = test_db.shuffle(60000)
train_db = train_db.batch(25)
test_db = test_db.batch(25)

##神经网络构建
network = Sequential([
    layers.Dense(100,activation='relu'),
    layers.Dense(50,activation='relu'),
    layers.Dense(20,activation='sigmoid'),
    layers.Dense(1)
])
network.build(input_shape=(25,113))##记得此处更改输入量大小
network.summary()
optimizer=tf.keras.optimizers.Adam(0.001)

##训练样本
j=0
for epoch in range(30):
    for step, (x, y) in enumerate(train_db):
        j = j+1
        with tf.GradientTape() as tape:
            out = network(x,training=True)
            loss = tf.keras.losses.mean_squared_error(y, out)
            grads = tape.gradient(loss, network.trainable_variables)
            optimizer.apply_gradients(zip(grads, network.trainable_variables))
    print('step:'+str(epoch))
print('已结束训练')

#对模型进行输出
output=network(x_caculate,training=False)
##对输出数据进行反标准化处理
output = scale_y.inverse_transform(output)
##将输出数据读取到excel中
output=pd.DataFrame(output)
writer = pd.ExcelWriter('')##读取excel表格的位置
output.to_excel(writer,float_format='%.5f')#todo_value指待倒入excel的数据，格式是dataframe
writer.save()

print('计算完毕')