import pandas as pd
import random


class Data:
    def __init__(self, data_path, train_rate=0.8, dependent_value=None, independent_value=None):
        self.data = None
        self.file_path = data_path
        self.train_rate = train_rate
        self.train_data = None
        self.test_data = None
        self.dependent_value = dependent_value
        self.independent_value = independent_value
        self.features = None  # 自变量（特征）
        self.target = None  # 因变量（目标变量）

    def read_excel_to_2d_array(self):
        # 读取CSV文件
        df = pd.read_csv(self.file_path)

        # 检查 dependent_value 是否在列名中
        if self.dependent_value not in df.columns:
            raise ValueError(f"'{self.dependent_value}' is not a valid column name in the data.")

        # 检查 independent_value 是否在列名中
        for col in self.independent_value:
            if col not in df.columns:
                raise ValueError(f"'{col}' is not a valid column name in the data.")

        # 将数据转换为二维列表并打乱顺序
        self.data = df.values.tolist()
        random.shuffle(self.data)

        # 提取目标变量和特征
        if self.dependent_value:
            target_index = df.columns.get_loc(self.dependent_value)  # 获取目标列的索引
            self.target = [row[target_index] for row in self.data]  # 提取目标列

        if self.independent_value:
            # 提取指定的特征列
            feature_index = [df.columns.get_loc(col) for col in self.independent_value]
            self.features = [[row[i] for i in feature_index] for row in self.data]
        else:
            raise ValueError(f"没有可用的feature或target")

    def print_data(self):
        if self.features:
            print("\nFeatures:")
            print(self.features)
        if self.target:
            print("\nTarget:")
            print(self.target)

    def ready_to_train(self):

        if(len(self.features) != len(self.target)):
            raise ValueError("变量长度不同，请检查")

        # 划分训练集和测试集
        train_size = int(len(self.features) * self.train_rate)
        self.train_data = (self.features[:train_size], self.target[:train_size])
        self.test_data = (self.features[train_size:], self.target[train_size:])


    def print_split_data(self):
        print("\nTraining Data:")
        print("Features:", self.train_data[0])
        if self.target:
            print("Target:", self.train_data[1])

        print("\nTesting Data:")
        print("Features:", self.test_data[0])
        if self.target:
            print("Target:", self.test_data[1])


# 示例用法
if __name__ == "__main__":
    # 实例化类并传入CSV文件路径
    allData = Data(
        "../../data/pokemon_go.csv",
        train_rate=0.8,
        dependent_value="cp",
        independent_value=["hp","weight"]  # 指定自变量列
    )

    # 读取CSV文件并转换为二维数组
    allData.read_excel_to_2d_array()

    # 准备训练数据
    allData.ready_to_train()

    # 打印数据
    allData.print_split_data()