import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

N = objCount = data.shape[0]
featuresCount = data.shape[1]

N_train = 413
N_test  = objCount - N_train

def normaliseData(data):
  for col in range(data.shape[1]):
    colMean = data[: ,col].mean()
    colStd = data[: , col].std()
    for row in range(data.shape[0]):
      if colStd != 0:
        data[row, col] = (data[row, col] - colMean) / colStd
  return data

def normaliseVector(vector):
  mean = vector.mean()
  std = vector.std()
  for i in range(len(vector)):
    vector[i] = (vector[i] - mean) / std
  return vector

def splitData(data, target):
  train_idx = random.sample(range(N), N_train)
  X_train = data[train_idx, :]
  T_train = target[train_idx]
  X_test  = np.delete(data, train_idx, axis = 0)
  T_test  = np.delete(target, train_idx, axis = 0)
  return X_train, T_train, X_test, T_test


dataNorm = normaliseData(data)

dataTrain, tTrain, dataTest, tTest = splitData(dataNorm, target)


def getMatrixPlan(X):
  N = X.shape[0]
  M = X.shape[1]
  F = np.ones((N, M + 1))
  F[:, 1:] = X
  return F

F_train = getMatrixPlan(dataTrain)

mu = 0
sigma = 0.01
w = np.random.normal(mu, np.sqrt(sigma), F_train.shape[1])

def gradient(F, t, w, alpha):
  t = np.array([t])
  w = np.array([w])
  a = -(t @ F)
  b = w @ (F.T @ F)
  c = alpha * w
  return (a + b + c).ravel()

def norma(vector):
  norma = 0
  for i in range(len(vector)):
    norma += vector[i] ** 2
  return np.sqrt(norma)

# train
Error = []
lr = 0.0001

eps = 0.0000000000000001

iter = 0
while iter != 1000:
  w_Old = w
  w = w - lr * gradient(F_train, tTrain, w, alpha = 0.1)
  if abs(norma(w_Old) - norma(w)) < eps:
    print("Near weights")
    break
  if norma(gradient(F_train, tTrain, w, alpha = 0.1)) < eps:
    print("Gradient is near to zero")
    break
  y_train = F_train @ w
  curError = 0
  for i in range(N_train):
    curError += (y_train[i] - tTrain[i])**2
  Error.append(curError / N_train)
  iter += 1

plt.plot(range(iter), Error)
plt.show()
print("Train error=", Error[-1])

F_test = getMatrixPlan(dataTest)
y_test = F_test @ w
curError = 0
for i in range(N_test):
  curError += (y_test[i] - tTest[i])**2
curError /= N_test
print("Test error=", curError)

