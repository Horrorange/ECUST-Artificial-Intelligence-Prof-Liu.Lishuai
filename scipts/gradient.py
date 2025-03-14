import numpy as np
from data import data

class Loss:

    def __init__(self):
        self.type = "Basic"
        self.loss = None

    def calculate_loss(self, data, parameters):
        self.loss = 0
        # train data is like [ [[1, 2], [3, 4]] , [3, 4]  ]
        ans = []



# 示例用法
if __name__ == "__main__":
    # 实例化类并传入CSV文件路径
    allData = data.Data(
        "../data/pokemon_go.csv",
        train_rate=0.8,
        dependent_value="cp",
        independent_value=["hp","weight"]  # 指定自变量列
    )
    allData.read_excel_to_2d_array()
    allData.ready_to_train()
    w = [[1, 2, 3],[2, 4, 1],[1, 8, 7]]
    # w is a matrix
    # y = w[0] 0*0 + w[1] hp + w[2] weight + w[3] hp^2 + w[4] weight^2 + w[5] weight * hp