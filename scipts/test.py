import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# def functions
def loss(num, b, w, data):
    ans = 0
    for i in range(num):
        ans += np.pow((data[i][1] - b - w * data[i][0]), 2) / num
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
testCount = dataCount - trainCount

# training and test data initialization

total_data = list(zip(cp, poweredCp))
random.shuffle(total_data)
train_data = total_data[:trainCount]
test_data = total_data[trainCount:]

print(test_data)

# fitting parameters
learningRate = 0.000013
w = 2
b = 2
epochs = 500
gradient = 0

# test codes
# print("Loss=", loss(trainCount, b, w, train_data) )
# print("Gradient= ", grad(trainCount, b, w, learningRate, train_data) )
# print("w=",w,", b=",b)

# fitting
for i in range(epochs):
    gradient = grad(
        trainCount, b, w, learningRate, train_data
    )
    if i % 30 == 0:
        print(
              "第",i + 1,"次的Loss为",
              loss(trainCount, b, w, train_data),
              "第",i + 1,"次的Gradient为",
              gradient
        )
    w -= learningRate * gradient[0]
    b -= learningRate * 10000 * gradient[1]
print("w=",w,", b=",b)

# testing_data_loss
print("Test data的loss为：",loss(testCount, b, w, test_data))

# plot
x = np.arange(0, 1000, 10)
y = w * x + b
plt.plot(x, y)
for i in range(testCount):
    plt.scatter(test_data[i][0], test_data[i][1])
plt.show()