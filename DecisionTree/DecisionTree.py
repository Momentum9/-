import pydotplus
from sklearn import tree
from IPython.display import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

# 加载iris数据
iris = datasets.load_iris()
X = iris.data[:, [0, 2]] # 只选取了两个特征
y = iris.target

# 训练模型，限制树的最大深度为4
clf = DecisionTreeClassifier(max_depth=4) # 默认使用gini coefficient
# 拟合模型 
clf.fit(X, y)

dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=['sepal length', 'sepal width'],
                                class_names=iris.target_names,
                                filled=True, rounded=True,
                                special_characters=True)
# feature_names 替换x0,x1
# filled 填充； rounded 圆角
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())