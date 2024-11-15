import numpy as np
from time import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import random
from scipy import stats
import pyqiopt as pq

#Загрузка дата сета
stock = np.loadtxt("task-1-stocks.csv",delimiter=",",dtype=float)

#Количество периодов наблюдений
n = 100
#Количество вспомогательных переменных
m = 20
#Размерность матрицы QUBO
N = m*n

#Вспомогательные функции для расчета матриц
def SS(stock_n:int, stock_j:int, period:int):
    sum = 0.0
    for l in range(n-1):
        sum += (stock[(0+period):(99+period),stock_n][l]*stock[(0+period):(99+period),stock_j][l])
    return sum

def S( stock_j:int, period:int):
    sum = 0.0
    for l in range(n-1):
        sum += (stock[(0+period):(99+period),stock_j][l])
    return sum
print(S(0,1))
#оптимизация S и SS 2**i
S0=np.zeros(n)
for j in range(n):
    S0[j]=S(j,0)
S1=np.zeros(n)
for j in range(n):
    S1[j]=S(j,1) 
SS0=np.zeros((n,n))
for j in range(n):
    for k in range(n):
        SS0[k,j]=SS(k,j,0)
SS1=np.zeros((n,n))
for j in range(n):
    for k in range(n):
        SS1[k,j]=SS(k,j,1)
two=np.zeros(m)
for i in range(m):
    two[i]=2**i

#Гамильтониан связанный с доходностью
def H2():
    H=np.zeros((N,N))
    for i in range(m):
        for j in range(n):
                    H[j*m+i,j*m+i] += (two[i]*(-stock[0:n,j][99]+stock[0:n,j][0]))                         
    return H/n
            
         
#Гамильтониан связанный с риском
def H1():
    H=np.zeros((N,N))
    
    for i in range(m):
        for j in range(n):
            H[j*m+i,j*m+i]+=0
    for i in range(m):
        for j in range(n):
            for l in range(m):
                for k in range(n):
                    H[j*m+i,k*m+l] += (two[i]*two[l]*SS1[k,j] - two[i]*two[l]*SS0[k,j] 
                                       - 1/n*two[i]*two[l]*S1[j]*S1[k] + 1/n*two[i]*two[l]*S0[j]*S0[k])  
                    
    return H* (n)/(n-1)

#Ограничение на покупку свыше 1кк
def H3():
    H=np.zeros((N,N))
    for i in range(m):
        for j in range(n):
            H[j*m+i,j*m+i]+=-2*1000000*two[i]*stock[0:n,j][0]
    for i in range(m):
        for j in range(n):
            for l in range(m):
                for k in range(n):
                    H[j*m+i,k*m+l] += two[i]*two[l]*stock[0:n,j][0]*stock[0:n,k][0]
    return H


#Константа сдвига
alpha = 20.00
#Итоговый гамильтониан
start1 = time()
HH1=H1()
HH2=H2()
HH3=H3()
quq=np.zeros((100,4))
for e in range(100):
    alpha = 10/100*i
    HQ=alpha*HH1+HH2+HH3
    start2 = time()
    sol = pq.solve(HQ, number_of_runs=1, number_of_steps=5000, return_samples=False, gpu=True)
    end2 = time();
    x = sol.vector
    end1 = time();
    #print("Algorithm end:", end1 - start1, "s")
    #print("Solver end:", end2 - start2, "s")
    print(e,"-----------\n")
    #Тестовые функции + классика
    #Средняя доходность
    def r_():
        sum3=0.0
        for l in range(n-1):
            sum1=0.0
            sum2=0.0
            for i in range(m):
                for j in range(n):
                    sum1+=x[j*m+i]*(2**i)*stock[0:n,j][l+1]
                    sum2+=x[j*m+i]*(2**i)*stock[0:n,j][l]
            sum3+=sum1/sum2
        return (sum3/n)-1

    #Доходность за период
    def rl(l:int):
        sum1=0.0
        sum2=0.0
        for i in range(m):
            for j in range(n):
                sum1+=x[j*m+i]*(2**i)*stock[0:n,j][l+1]
                sum2+=x[j*m+i]*(2**i)*stock[0:n,j][l]
        return (sum1/sum2)-1

    #Риск портфеля за весь период
    def sigma():
        result = 0.0
        r=r_()
        for l in range(n-1):
            result+=(rl(l)-r)**2
        return(np.sqrt(n/(n-1)*result))

    #Стоимость портфеля в начале
    def cost_start():
        sum1=0.0
        for i in range(m):
            for j in range(n):
                sum1+=x[j*m+i]*(2**i)*stock[0:n,j][0]
        return sum1

    #Стоимость портфеля в конце
    def cost_end():
        sum1=0.0
        for i in range(m):
            for j in range(n):
                sum1+=x[j*m+i]*(2**i)*stock[0:n,j][n-1]
        return sum1

    #print("Average profit: ", r_())
    print("Average risk: ", sigma())
    print("Start cost: ", cost_start())
    #print("Start end: ", cost_end())
    print("End profit: ", cost_end()/cost_start() - 1)
    print("Alpha: ", alpha)
    quq[e]=[alpha,cost_start(),sigma(),cost_end()/cost_start() - 1]

np.savetxt("quq.txt",quq,delimiter=",")