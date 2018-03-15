from sklearn import datasets
import numpy as np


def getTrainX():
    iris = datasets.load_iris()
    return np.append(iris.data[0:40], iris.data[50:90], axis=0)


def getTrainY():
    iris = datasets.load_iris()
    a = np.array([[1]])
    # loc = 1
    # for i in iris.target:
    #     b = np.array([int(i / 2), i % 2])
    #     a = np.row_stack((a, b))
    #     loc = loc + 1
    # a = np.delete(a, 0, 0)
    for i in np.append(iris.target[0:40], iris.target[50:90]):
        b = np.array([i])
        a = np.row_stack((a, b))
    return np.delete(a, 0, 0)


def getTestX():
    iris = datasets.load_iris()
    return np.append(iris.data[40:50], iris.data[90:100], axis=0)


def getTestY():
    iris = datasets.load_iris()
    a = np.array([[1]])
    for i in np.append(iris.target[40:50], iris.target[90:100]):
        b = np.array([i])
        a = np.row_stack((a, b))
    return np.delete(a, 0, 0)
