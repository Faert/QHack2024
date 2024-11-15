import numpy as np
from time import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import random
from scipy import stats
#import pyqiopt as pq

#Загрузка дата сета
stock = np.loadtxt("task-1-stocks.csv",delimiter=",",dtype=float)

#Количество периодов наблюдений
n = 100
#Количество вспомогательных переменных
m = 20
#Размерность матрицы QUBO
N = m*n

#Вспомогательные функции для расчета матриц

def cost_end2( sn:int ):
    count=1000000//stock[0:n,sn][0]
    return stock[0:n,sn][99]*count        
    
def cost_start2(sn:int):
    count=1000000//stock[0:n,sn][0]
    return stock[0:n,sn][0]*count      

def r_2(sn:int):
    count=1000000//stock[0:n,sn][0]
    sum=0.0
    for l in range(n-1):
        sum+=(stock[0:n,sn][l+1]*count)/(stock[0:n,sn][l]*count)-1
    return sum/n

#Доходность за период
def rl2(l:int,sn:int):
    count=1000000//stock[0:n,sn][0]
    return (stock[0:n,sn][l+1]*count)/(stock[0:n,sn][l]*count)-1

#Риск портфеля за весь период
def sigma2(sn:int):
    result = 0.0
    r=r_2(sn)
    for l in range(n-1):
        result+=(rl2(l,sn)-r)**2
    return(np.sqrt(n/(n-1)*result))

for e in range(n-1):
    if((cost_end2(e)/cost_start2(e) - 1)>0.3):
        print("-", e ,"  ", r_2(e),"  ",sigma2(e), "  ",cost_end2(e)/cost_start2(e) - 1)

