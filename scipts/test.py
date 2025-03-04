import pandas as pd
import numpy as np

# def functions
def loss(n, y_predicted, b, w, x):
    return np.sum(np.pow((y_predicted - b - w * x), 2)) / n

def grad(n, y_predicted, b, w, x, lr):
    return [
        (-loss(n, y_predicted, b, w, x) + loss(n, y_predicted, b, w + lr, x)) / lr ,
        (-loss(n, y_predicted, b, w ,x) + loss(n, y_predicted, b + lr, w ,x)) / lr
    ]

# read the data
path = "../data/pokemon_go.csv"
data = pd.read_csv(path)
cp = data[['cp']]
poweredCp = data[['cp_new']]

# basic parameters
dataCount = len(data)
trainPercent = 0.8
testPercent = 1 - trainPercent

# training and test data initialization
trainCp = cp[:int(dataCount * trainPercent)].values
testCp = cp[int(dataCount * trainPercent):].values
trainPoweredCp = poweredCp[:int(dataCount * trainPercent)].values
testPoweredCp = poweredCp[int(dataCount * trainPercent):].values

# fitting parameters
learningRate = 0.00002
w = 2
b = 2
epochs = 500
gradient = 0

print("Loss=", loss(int(dataCount * trainPercent), trainPoweredCp, b, w, trainCp))
print("Gradient= ", grad(int(dataCount * trainPercent), trainPoweredCp, b, w, trainCp, learningRate) )
print("w=",w,", b=",b)

# fitting
for i in range(epochs):
    gradient = grad(
        dataCount * trainPercent, trainPoweredCp, b, w, trainCp, learningRate
    )
    print(
          "第",i,"次的Loss为",
          loss(int(dataCount * trainPercent) , trainPoweredCp, b, w , trainCp),
          "第",i,"次的Gradient为",
          gradient
    )
    w -= learningRate * gradient[0]
    b -= learningRate * 10000 * gradient[1]
print("w=",w,", b=",b)
