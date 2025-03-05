import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# def functions
def loss(w, data):
    ans = 0
    rms = 0
    for i in range(len(data)):
        ans = data[i][1]
        for j in range(len(w)):
            ans -= w[j] * np.pow(data[i][0] , j)
        rms += pow(ans, 2)/len(data)
    return rms

def grad(w, lr, data):
    ans = []
    loss_ini = loss(w, data)
    for i in range(len(w)):
        w[i] += lr
        ans.append(loss(w, data) - loss_ini)
        w[i] -= lr
    return ans

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

# fitting parameters
learningRate = 0.000013
learningRateMulti = [1000, 1000, 0.12]
w = [0, 0, 0]
epochs = 1000
gradient = 0

# test codes
print("Loss=", loss(w, train_data) )
print("Gradient= ", grad(w, learningRate, train_data) )


# fitting
for i in range(epochs):
    gradient = grad(
        w, learningRate, train_data
    )
    if i % 20 == 0:
        print(
              "第",i + 1,"次的Loss为",
              loss(w, train_data),
              "第",i + 1,"次的Gradient为",
              gradient
        )
    for i in range(len(w)):
        w[i] -= learningRate * learningRateMulti[i] * gradient[i]

# testing_data_loss
print("Test data的loss为：",loss(w, test_data))
print("所求得的参数为：",w)
# plot
x = np.arange(0, 800, 10)
y = x.copy()
for i in range(len(x)):
    y[i] *= 0
    for j in range(len(w)):
        y[i] += w[j] * pow(x[i], j)
plt.plot(x, y)
# for i in range(trainCount):
#     plt.scatter(train_data[i][0], train_data[i][1])
for i in range(testCount):
    plt.scatter(test_data[i][0], test_data[i][1])
plt.show()