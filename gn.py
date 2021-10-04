# 이이균 작성

import numpy as np
from scipy.misc import derivative
import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/20708038/scipy-misc-derivative-for-multiple-argument-function 
def partial_derivative(func, var=0, point=[]):
    args = point[:] 
    def wraps(x):
        args[var] = x
        return func(*args)
    return derivative(wraps, point[var], dx = 1e-6)

def linear(x,b): # 회귀모델 사용하였음
  return b[0] + (x * b[1]) + ((x * x) * b[2]) + ((x * x * x) * b[3]) # x + ax + bx^2

def get_residual_mat(x,y,b,f): # 잔차 값 배열 생성
  err = []
  for ib in range(0,len(b)):
    err.append([])
    for i in range(0,len(x)):
      err[ib].append(y[i] - f(x[i],b))

  return err

def get_residual(x,y,b): # 잔차 값 생성
  return y - (b[0] + (x * b[1])  + ((x * x) * b[2]) + ((x * x * x) * b[3])) # x + ax + bx^2

def get_jaco(x,y,b,f): # 잔차식 야코비안 생성
  h = []
  for ib in range(0,len(b)):
    h.append([])
    for i in range(0,len(x)):
      h[ib].append(partial_derivative(get_residual,2,[x[i],y[i],b]))

  return h

def gn(g,x,y,b0,iter = 100):
  nb = b0
  x_pt = []
  y_pt = []
  for i in range(0, iter):
        h = np.array(get_jaco(x,y,nb,g))
        r = np.array(get_residual_mat(x,y,nb,g))

        sd = np.linalg.pinv(h)
        vgn = np.matmul(r,sd)
        nb_mat = np.subtract(nb,vgn)
        nb = nb_mat[0]

        #x_pt.append(x[i]) # Fit line X
        #ny = g(x[i],nb) # get update value
        #y_pt.append(ny) # Fit line Y

  return nb#x_pt,y_pt;

def draw_fit(g,x,y,nb): # fit 된 라인 포인트를 반환
  x_pt = []
  y_pt = []
  for i in range(0, len(x)):
      x_pt.append(x[i]) # Fit line X
      ny = g(x[i],nb) # get update value
      y_pt.append(ny) # Fit line Y

  return x_pt,y_pt

np.random.seed(2)
b0 = np.array([2,1.5,1.2,1.4]); # 초기 파라메터
st_dev = 40
loc = 200
x = np.random.normal(3, 1, 100)
y = linear(np.random.normal(loc, st_dev, 100),b0)# / x
nb = gn(linear,x,y,b0);
x_pt,y_pt = draw_fit(linear,x,y,nb);
plt.scatter(x_pt, y_pt, color='red')
plt.scatter(x, y)
plt.show()
