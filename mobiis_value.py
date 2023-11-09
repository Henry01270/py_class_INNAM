import numpy as np
import pandas as pd
import tensorflow as tf

dp = pd.read_csv('mobiis_price.csv') #데이터프레임 불러오기

data = []
x데이터 = []
y데이터 = []

for i in range(0, len(data)):
    data.append(dp.iat[i, 3])

for i in range(0, len(data) - 25):
    x데이터.append(data[i : i + 25])
    y데이터.append(data[i + 25])

x데이터 = tf.one_hot(x데이터, 31)
y데이터 = tf.one_hot(y데이터, 31)

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(100, input_shape = (25, 31)),
    tf.keras.layers.Dense(31, activation = 'softmax')
])

model.compile(loss ='categorical_crossentropy', optimizer = 'adam')

model.fit(x데이터, y데이터, batch_size=64, epochs = 30, verbose = 2)



#predata = [2600,2570,2450,2420,2555,2640,2705,3970,4900,6850,6230,5960,5830,5220,5350,4750,4910,4890,5260,5220,4945,5290,6870,6630,5850]
