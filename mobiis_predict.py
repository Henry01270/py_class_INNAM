import numpy as np
import pandas as pd
import tensorflow as tf

dp = pd.read_csv('mobiis_price.csv') #데이터프레임 불러오기

data = {}
x = []
x데이터 = []
y데이터 = []

for i in range(0, 493):
    data[i] = dp.iat[i, 3]

for i in range(0, 493):
    x.append(data[i])

for i in range(0, len(x) - 25):
    x데이터.append(x[i : i + 25])
    y데이터.append(x[i + 25])

x데이터 = tf.one_hot(x데이터, 1067) #min: 1540 #max: 6870
y데이터 = tf.one_hot(y데이터, 1067)

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(100, input_shape = (25, 1067)),
    tf.keras.layers.Dense(1067, activation = 'softmax')
])

model.compile(loss ='categorical_crossentropy', optimizer = 'adam')

model.fit(x데이터, y데이터, batch_size=64, epochs = 300)


predata = [2600,2570,2450,2420,2555,2640,2705,3970,4900,6850,6230,5960,5830,5220,5350,4750,4910,4890,5260,5220,4945,5290,6870,6630,5850]
predata = tf.one_hot(predata, 1067)
predata = tf.expand_dims(predata, axis = 0)

predict = model.predict(predata)
predict = np.argmax(predict[0])
print(predict * 5 + 1540)