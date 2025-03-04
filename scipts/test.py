import pandas as pd
import numpy as np
import random

# def functions
def loss(num, b, w, train_data):
    ans = 0
    for i in range(trainCount):
        ans += np.pow((train_data[i][1] - b - w * train_data[i][0]), 2) / num
    return ans

def grad(num, b, w, lr, train_data):
    return [
        (-loss(num, b, w, train_data) + loss(num, b, w + lr, train_data)) / lr ,
        (-loss(num, b, w, train_data) + loss(num, b + lr, w, train_data)) / lr
    ]

# read the data
path = "../data/pokemon_go.csv"
data = pd.read_csv(path)
cp = data['cp'].tolist()
poweredCp = data['cp_new'].tolist()


# basic parameters
dataCount = len(data)
trainPercent = 0.8
trainCount = int(dataCount * trainPercent)

# training and test data initialization

total_data = list(zip(cp, poweredCp))
random.shuffle(total_data)
train_data = total_data[:trainCount]
test_data = total_data[trainCount:]

# fitting parameters
learningRate = 0.000015
w = 2
b = 2
epochs = 500
gradient = 0

print("Loss=", loss(trainCount, b, w, train_data) )
print("Gradient= ", grad(trainCount, b, w, learningRate, train_data) )
print("w=",w,", b=",b)

# fitting
for i in range(epochs):
    gradient = grad(
        trainCount, b, w, learningRate, train_data
    )
    print(
          "第",i,"次的Loss为",
          loss(trainCount, b, w, train_data),
          "第",i,"次的Gradient为",
          gradient
    )
    w -= learningRate * gradient[0]
    b -= learningRate * 10000 * gradient[1]
print("w=",w,", b=",b)