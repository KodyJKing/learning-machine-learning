# A very basic RNN that just learns to sum a sequence.

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
import random

SEQ_LENGTH = 100

def getSeqSum(n):
    x = []
    y = []
    for _ in range(n):
        seq = []
        sum = 0
        for i in range(SEQ_LENGTH):
            v = random.randint(0, 100)
            sum += v
            seq.append(v)
        x.append(seq)
        y.append(sum)
    return np.array(x), np.array(y)

def getData():
    filename = "data/sequence_sum.npz"
    # if (not os.path.exists(filename)):
    x, y = getSeqSum(1000)
    x_test, y_test = getSeqSum(300)
    np.savez_compressed(filename, x, y, x_test, y_test)
    return np.load(filename)

# print(x)

model = models.Sequential([
    layers.Input((SEQ_LENGTH, 1)),
    layers.SimpleRNN( 10, activation="relu" ),
    layers.Dense(1, activation="relu")
])
model.summary()
model.compile(loss="mean_squared_error", optimizer="adam")

with getData() as data:
    x, y, x_test, y_test = data.values()

    x = np.reshape(x, (len(y), SEQ_LENGTH, 1))
    x_test = np.reshape(x_test, (len(y_test), SEQ_LENGTH, 1))

    model.fit(x, y, epochs=20)

    y_predict = model.predict(x_test)

    avgRatio = 0
    count = len(y_predict)
    for i in range(count):
        predicted = y_predict[i][0]
        actual = y_test[i]
        ratio = predicted / actual if predicted > actual else actual / predicted
        avgRatio += ratio / count
        print( predicted, actual, ratio )
    print("Average ratio:", avgRatio)
