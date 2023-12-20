from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
import numpy as np
import random


digits = load_digits()
X = digits.data
Images = digits.images
Target = digits.target
NumClasses = len(digits.target_names)
N = objCount = X.shape[0]

N_train = N // 3 * 2
N_val  = N - N_train

def normaliseData(data):
    for col in range(data.shape[1]):
        colMean = data[: ,col].mean()
        colStd = data[: , col].std()
        for row in range(data.shape[0]):
            if colStd != 0:
              data[row, col] = (data[row, col] - colMean) / colStd
    return data



def splitData(data, target):
    train_idx = random.sample(range(N), N_train)
    X_train = data[train_idx, :]
    T_train = target[train_idx]
    Images_train = Images[train_idx, :]
    X_val  = np.delete(data, train_idx, axis = 0)
    T_val  = np.delete(target, train_idx, axis = 0)
    Images_val = np.delete(Images, train_idx, axis = 0)
    return X_train, T_train, X_val, T_val, Images_train, Images_val

def oneHotEncoding(Target):
    OneHotTarget = np.zeros([Target.shape[0], NumClasses])
    curIdx = 0
    for num in Target:
        OneHotTarget[curIdx, num] = 1
        curIdx += 1
    return OneHotTarget

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

    
def SoftMax(x, w, bias):
    x = np.atleast_2d(x) @ w.T
    resultMatrix = np.zeros_like(x)
    for row in range(x.shape[0]):
        znamenatel = 0
        for elem in x[row]:
            znamenatel += np.exp(elem)
        for col in range(len(resultMatrix[row])):
            resultMatrix[row, col] = np.exp(x[row, col]) / znamenatel
    return resultMatrix



def Error(x, w, t, bias, _lambda = 0.01):
    N = x.shape[0]
    sum = 0
    for i in range(N):
          for c in range(NumClasses):
              y = SoftMax(x[i], w, bias)
              sum += t[i, c] * np.log(y[0, c])
    sum += (_lambda / 2) + np.sum(w ** 2)
    return -sum / N

def gradient(x, w, t, bias, _lamda = 0.01):
    N = x.shape[0]
    sum = np.zeros([10, 64])
    for i in range(N):
          sum += (SoftMax(x[i], w, bias) - np.atleast_2d(t[i])).T @ np.atleast_2d(x[i])
    return sum + _lamda * w

def gradient_bias(x, w, t, bias):
    N = x.shape[0]
    sum = np.zeros([1, 10])
    for i in range(N):
          sum += (SoftMax(x[i], w, bias) - np.atleast_2d(t[i]))
    return sum


def norma(matrix):
    norma = 0
    for row in matrix:
        for elem in row:
            norma += elem ** 2
    return np.sqrt(norma)

def confusionMatrix(x, w, bias, target):
      matrix = np.zeros([NumClasses, NumClasses])
      for rowIdx in range(len(x)):
          curTarget = SoftMax(x[rowIdx], w, bias)
          curTarget = np.argmax(curTarget)
          matrix[np.argmax(target[rowIdx]), curTarget] += 1
      return matrix

def accuracy(confusionMatrix):
      sum = 0
      n = len(confusionMatrix)
      for i in range(n):
          sum += confusionMatrix[i, i]
      return sum / np.sum(confusionMatrix)


X_normalise = normaliseData(X)
OneHotTarget = oneHotEncoding(Target)

X_train, T_train, X_val, T_val, Images_train, Images_val = splitData(X_normalise, OneHotTarget)

fig, axs = plt.subplots(2, 2)
fig.set_size_inches(14, 7)

class LogisticModel:
  def __init__(self, weights = np.random.normal(0, 0.001, size = (10, 64)), bias = np.random.normal(0, 0.001, size = (1, 10))):   
    self.w = weights
    self.b = bias
  def fit(self, x, t):
    print("Confusion matrix before fit: \n")
    ConfusionMatrix = confusionMatrix(X_val, self.w, self.b, T_val)
    print(ConfusionMatrix)
    print("Accuracy before fit: \n")
    print(accuracy(ConfusionMatrix))
    X_train = x
    T_train = t
    iterNum = 10
    ErrorListVal = []
    ErrorListTrain = []
    AccuracyListVal = []
    AccuracyListTrain = []
    lr = 0.001
    eps = 0.0000001 

    for iter in range(iterNum):
      w_Old = self.w
      self.w = self.w - lr * gradient(X_train, w_Old, T_train, self.b)
      self.b = self.b - lr * gradient_bias(X_train, w_Old, T_train, self.b)
      y_train = SoftMax(X_train, self.w, self.b)
      ErrorListTrain.append(Error(X_train, self.w, T_train, self.b))
      ErrorListVal.append(Error(X_val, self.w, T_val, self.b))
      AccuracyListTrain.append(accuracy(confusionMatrix(X_train, self.w, self.b, T_train)))
      AccuracyListVal.append(accuracy(confusionMatrix(X_val, self.w, self.b, T_val)))
      if iter % 5 == 0:
        print(f'iter: {iter}')
        print(f'Accuracy Train: {AccuracyListTrain[-1]}')
        print(f'Целевая Train: {ErrorListTrain[-1]}')
        print(f'Accuracy Val: {AccuracyListVal[-1]}')
        print(f'Целевая Val: {ErrorListVal[-1]}')

    axs[0, 0].plot(range(iter + 1), ErrorListVal)
    axs[0, 1].plot(range(iter + 1), AccuracyListVal)
    axs[0, 0].set_xlabel("Целевая Validation")
    axs[0, 1].set_xlabel("Accuracy Validation")
    axs[1, 0].plot(range(iter + 1), ErrorListTrain)
    axs[1, 1].plot(range(iter + 1), AccuracyListTrain)
    axs[1, 0].set_xlabel("Целевая Train")
    axs[1, 1].set_xlabel("Accuracy Train")

    plt.show()
    print("Confusion matrix after fit: \n")
    ConfusionMatrix = confusionMatrix(X_val, self.w, self.b, T_val)
    print(ConfusionMatrix)
    print("Accuracy after fit: \n")
    print(accuracy(ConfusionMatrix))

  def predict(self, x):
    print(np.argmax(SoftMax(x, self.w, self.b)))
      
model = LogisticModel()
model.fit(X_train, T_train)



