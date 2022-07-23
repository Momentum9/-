import numpy as np
from collections import Counter


class KNearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        """
        KNN算法的训练方法很简单，仅仅是把图像都存储下来
        Args:
            X (np.array): 训练集的特征 shape(num_train, dimension) # 对于图像(32,32,3) dimension为32*32*3
            y (np.array): 训练集的类别 shape(num_train,)
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1):
        """预测测试集
        Args:
            X (np.array): 测试集的特征 shape(num_test, demension)
            k (int, optional): 默认为最近邻算法. Defaults to 1.

        Returns:
            y_pred: 测试集的类别 shape(num_test,)
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        Y_pred = np.zeros(num_test)

        # 注：求dists还可通过更为彻底的广播机制和矩阵乘法实现
        for i in range(num_test):
            dists[i] = np.sum((self.X_train - X[i]) ** 2,axis=1) ** 0.5  # L2距离; 按行求和
            closest_y = []
            # 根据dists找到与第i个测试样本距离最近的k个训练样本,并将其标签存入closest_y
            idx = np.argsort(dists[i])  # [第一小索引,第二小索引,...]
            closest_y = [self.y_train[j] for j in idx][:k]  # 取出该列表生成器的前k个

            # 找到k个类别中频次最多的类别
            c = Counter(closest_y)
            Y_pred[i] = c.most_common(1)[0][0]

        return Y_pred

    def score(self, X, y, k=1):
        """评测准确度
        Args:
            X : 测试集的特征
            y : 测试集的已知类别
            k (int, optional):  Defaults to 1.
        """
        Y_pred = self.predict(X,k)
        num_correct = np.sum(Y_pred == y)
        accuracy = float(num_correct) / len(X)
        return accuracy
