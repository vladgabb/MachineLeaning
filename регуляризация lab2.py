import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import sys

N = 1000
x = np.linspace(0, 1, N)
z = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
error = 10 * np.random.randn(N)
t = z + error


N_train = int(0.9 * N)
N_val = int(0.09 * N)
N_test = int(0.01 * N)

def DataSplit(x) -> tuple:
  train_idx = random.sample(range(N), N_train)
  X_train = x[train_idx]
  T_train = t[train_idx]


  x_test = np.delete(x, train_idx)
  t_test = np.delete(t, train_idx)
  test_idx = random.sample(range(N - len(X_train)), N_test)
  X_test = x_test[test_idx]
  T_test = t_test[test_idx]


  x_val = np.delete(x_test, test_idx)
  t_val = np.delete(t_test, test_idx)
  val_idx = random.sample(range(N - len(X_train) - len(X_test)), N_val)
  X_val = x_val[val_idx]
  T_val = t_val[val_idx]

  return X_train, X_val, X_test, T_train, T_val, T_test, train_idx, test_idx, val_idx

X_train, X_val, X_test, T_train, T_val, T_test, train_idx, test_idx, val_idx = DataSplit(x)

maxPower = 20
M_baseFunctions = 8

baseFunctions = list(np.arange(maxPower + 1)) # степени полиномов
baseFunctions.append(lambda arg: np.sin(arg))
baseFunctions.append(lambda arg: np.cos(arg))
baseFunctions.append(lambda arg: np.tan(arg))
baseFunctions.append(lambda arg: np.sqrt(arg))
baseFunctions.append(lambda arg: np.exp(arg))
baseFunctions.append(lambda arg: np.exp(arg) ** 2)



def get_plan(baseFunctionsIdx, X):
  F = np.ones([X.shape[0], M_baseFunctions])
  i = 0
  iOnes = 0
  onesIdx = 0
  for idx in baseFunctionsIdx:
    if idx == 0:
      iOnes = i
    if 1 <= idx <= maxPower + 1:
      F[:, i] = X ** idx
    elif idx > maxPower:
      F[:, i] = baseFunctions[idx](X)  
    i += 1
  F[:, iOnes] = 1
  return F

def get_weights(F, t, alpha):
  w = np.linalg.inv((np.transpose(F) @ F) + alpha * np.ones((F.shape[1], F.shape[1]))) @ np.transpose(F) @ t
  return w



bestBaseFunctionsIdx = []
bestAlpha = -10
bestError = float('inf') 
best_w = 0
iter = 0


# Train
for baseFunctionsChoise in range(70):
  baseFunctionsIdx = random.sample(range(len(baseFunctions)), M_baseFunctions)
  F_train = get_plan(baseFunctionsIdx, X_train)
  for alpha in [0.01, 0.001, 0.0001, 0.00001, 0.000001]:
    w_train = get_weights(F_train, T_train, alpha)
    F_validation = get_plan(baseFunctionsIdx, X_val) 
    yVal = F_validation @ w_train # Валидация
    curErr = 0
    for i in range(len(yVal)):
      curErr += (yVal[i] - T_val[i]) ** 2
    curErr = curErr / int(N_val) 
    if curErr <= bestError:
      bestError = curErr
      bestBaseFunctionsIdx = copy.deepcopy(baseFunctionsIdx)
      bestAlpha = alpha
      best_w = w_train
    iter += 1




listStrBase = ["sin", "cos", "tan", "sqrt", "exp", "exp^2"]
baseFunctions.append(lambda arg: np.sin(arg))
baseFunctions.append(lambda arg: np.cos(arg))
baseFunctions.append(lambda arg: np.tan(arg))
baseFunctions.append(lambda arg: np.sqrt(arg))
baseFunctions.append(lambda arg: np.exp(arg))
baseFunctions.append(lambda arg: np.exp(arg) ** 2)

F = get_plan(bestBaseFunctionsIdx, x)
y = F @ best_w
fig = plt.figure(figsize=(8,6))
plt.plot(x, t, 'go')
plt.plot(x, z, 'r')
plt.plot(x, y, 'b')
print(f"Лучший коэффициент регуляризации a = {bestAlpha}")
print("Базисные ф-ии: ", end = " ")
for idx in bestBaseFunctionsIdx:
  if idx <= maxPower:
    print(f"x^{idx}", end = " ")
  else:
    print(listStrBase[idx % maxPower], end = " ")
curErr = 0
for i in range(N):
  curErr += (y[i] - z[i]) ** 2
curErr = curErr / int(N)
print("\n")
print(f"Error test = {curErr}")
plt.show()