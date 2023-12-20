import numpy as np
import matplotlib.pyplot as plt
# <  T это класс 0
# >= T это класс 1
# class 0 - футболисты
# class 1 - баскетболисты
NumElems = 1000


mu_footballers    = 180
sigma_footballers = 5
footballers = np.random.normal(mu_footballers , sigma_footballers, NumElems) # class 0

mu_basketballers    = 200
sigma_basketballers = 10
basketballers = np.random.normal(mu_basketballers , sigma_basketballers, NumElems) # class 1

AllClassesNumElems = len(basketballers) + len(footballers)


def TP(T):
  TP = 0
  for player in basketballers:
    if player >= T:
      TP += 1
  return TP

def TN(T):
  TN = 0
  for player in footballers:
    if player < T:
      TN += 1
  return TN

def FP(T):
  FP = 0
  for player in footballers:
    if player >= T:
      FP += 1
  return FP

def FN(T):
  FN = 0
  for player in basketballers:
    if player < T:
      FN += 1
  return FN

def Accuracy(T):
  return (TP(T) + TN(T)) / AllClassesNumElems

def Precision(T):
  if TP(T) + FP(T) == 0:
      return 0
  return (TP(T)) / (TP(T) + FP(T))

def Recall(T):
  return (TP(T)) / (TP(T) + FN(T))

def F1Score(T):
  if Precision(T) + Recall(T) == 0:
      return 0
  return 2 * (Precision(T) * Recall(T)) / (Precision(T) + Recall(T))

def Alpha(T):
  return FP(T) / (TN(T) + FP(T))

def Beta(T):
  return FN(T) / (TP(T) + FN(T))

def TPR(T):
  return TP(T) / (TP(T) + FN(T))

def FPR(T):
  return FP(T) / (FP(T) + TN(T))

def TPR_Step(T):
  return 1 / (TP(T) + FN(T))

def FPR_Stpe(T):
  return 1 / (FP(T) + TN(T))

def ROC_Coords(T):
  xCoords = [0, 0, 1]
  yCoords = [0, 0, 1] 
  xCoords[1] = FPR(T)
  yCoords[1] = TPR(T)
  return xCoords, yCoords

def ROC_Area(TPRList, FPRList):
  # x, y = ROC_Coords(T)
  # dx1 = x[1] - x[0]
  # dy1 = y[1] - y[0]
  # dx2 = x[-1] - x[1]
  # dy2 = y[-1] - y[1]
  # if dy1 == 1 and dx1 == 1:
  #   return 1/2
  # else:
  #   TriangleSquare1 = 1/2 * dx1 * dy1
  #   TriangleSquare2 = 1/2 * dx2 * dy2
  #   SideRectangleA = dx2
  #   SideRectangelB = dy1
  #   SquareRectangle =  SideRectangleA * SideRectangelB
  #   return TriangleSquare1 + SquareRectangle + TriangleSquare2
  Area = 0
  for i in range(len(TPRList) - 1):
    x_i = TPRList[i]
    x_iplus1 = TPRList[i+1]
    dx = x_iplus1 - x_i
    y_i = FPRList[i]
    Area += dx * y_i
  return Area

FPRList = []
TPRList = []
for T in range(0, 220):
  FPRList.append(TPR(T))
  TPRList.append(FPR(T))
TPRList = TPRList[::-1]
FPRList = FPRList[::-1]
Area = ROC_Area(TPRList, FPRList)

plt.plot(TPRList, FPRList)
plt.xlabel("TPR")
plt.ylabel("FPR")
plt.show()
print(Area)

#Accuracy и подсчитать для него основные метрики: TP, TN, FP, FN, Accuracy, Precision, Recall, F1-score, ошибки 1-го и 2-го рода (alpha, beta)
maxAccuracy = 0
for T in range(0, 220):
  curAccuracy = Accuracy(T)
  if curAccuracy > maxAccuracy:
    maxAccuracy = curAccuracy
    bestT  = T

x, y = ROC_Coords(bestT)
print("Best T:", bestT)
print(f"TP={TP(bestT)}, TN={TN(bestT)}, FP={FP(bestT)}, FN={FN(bestT)}, Accuracy={Accuracy(bestT)}")
print(f"Precision={Precision(bestT)}, Recall={Recall(bestT)}, F1-score={F1Score(bestT)}, Alpha={Alpha(bestT)}, Beta={Beta(bestT)}")
