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
        rms += pow(ans, 2) / len(data)
    return rms

def grad(w, lr, data):
    ans = []
    loss_ini = loss(w, data)
    for i in range(len(w)):
        w[i] += lr * 0.001
        ans.append(loss(w, data) - loss_ini)
        w[i] -= lr * 0.001
    return ans

class Parameter:
    def __init__(self):
        self.w = []

    def initialization(self, dimension, min, max):
        self.w  = [random.uniform(min, max) for i in range(dimension)]
        return self.w

class ProcessedData:
    def __init__(self, independent_variable, dependent_variable, training_rate):
        if len(independent_variable) != len(dependent_variable):
            raise ValueError("independentVariable and dependentVariable should have same length")
        self.dataCount = len(independent_variable)
        self.trainCount = int(self.dataCount * training_rate)
        self.testCount = int(self.dataCount - self.trainCount)

        self.total_data = list(zip(independent_variable, dependent_variable))
        random.shuffle(self.total_data)
        self.train_data = self.total_data[0:self.trainCount]
        self.test_data = self.total_data[self.trainCount:self.dataCount]

    def show_loss(self, w):
        print("Loss= ",loss(w, self.train_data))

    def show_grad(self, w, learning_rate):
        print("Grad=", grad(w, learning_rate, self.train_data))

    def regression_fitting_show_procedure(self, epochs, w, learning_rate):
        for i in range(epochs):
            gradient = grad(
                w, learningRate, data.train_data
            )
            if (i + 1) % 20 == 0:
                print(
                    "第", i + 1, "次的Loss为",
                    loss(w, data.train_data),
                    "第", i + 1, "次的Gradient为",
                    gradient
                )
            for i in range(len(w)):
                w[i] -= learningRate * learningRateMulti[i] * gradient[i]
        return w

    def regression_fitting(self, epochs, w, learning_rate):
        for i in range(epochs):
            gradient = grad(
                w, learningRate, data.train_data
            )
            for i in range(len(w)):
                w[i] -= learningRate * learningRateMulti[i] * gradient[i]
        return w

    def plotting(self, w):
        x = np.arange(0, 800, 10)
        y = x.copy()
        for i in range(len(x)):
            y[i] *= 0
            for j in range(len(w)):
                y[i] += w[j] * pow(x[i], j)
        plt.plot(x, y)
        # for i in range(data.trainCount):
        #     plt.scatter(data.train_data[i][0], data.train_data[i][1])
        for i in range(self.testCount):
            plt.scatter(self.test_data[i][0], self.test_data[i][1])
        # plt.show()

# read the data
path = "../data/pokemon_go.csv"
allData = pd.read_csv(path)
cp = allData['cp'].tolist()
poweredCp = allData['cp_new'].tolist()

for i in range(10):
    # data preprocess
    data = ProcessedData(cp, poweredCp, 0.85)


    # fitting parameters
    learningRate = 0.0013
    learningRateMulti = [10, 10, 0.047]
    para = Parameter()
    w = Parameter.initialization(para, 3, -20, 20)


    data.show_loss(w)
    data.show_grad(w,learningRate)

    w = data.regression_fitting(1000, w, learningRate)
    print(w)


    # testing_data_loss
    print("Test data的loss为：",loss(w, data.test_data))
    print("所求得的参数为：",w)

    data.plotting(w)
plt.show()