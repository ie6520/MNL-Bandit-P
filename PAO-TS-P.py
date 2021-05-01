
import numpy as np
from numpy.lib.function_base import place
import Optimal_Assortment
import matplotlib.pyplot as plt

from sample import Generate_theta
from sample import Generate_theta_p

import json

#feature dimension
D = 2
L = 3
#item set
N = [1,2,3,4,5,6,7,8]
#cardinality constraint
B = 5

r = [8,7,6,5,4,3,2,1]
#2d-array
Theta_g=(2*np.random.normal(0,1.0,size=(D,L))-np.random.uniform(size = (D,L)))/np.sqrt(D)
b_g = np.random.normal(0,1.0,size = len(N))

prod_f = np.random.normal(0,1.0,size = (len(N),L))

#print(p)
#print(b)

#print(Theta)
#print(p[0])
#print(np.matmul(Theta,p[0]))

def Receive_x():
    return np.random.rand(D)

def Prod(x):
    N_x = []
    return N

def getUtility(Theta,b,item,x):
    return np.dot(np.matmul(Theta,prod_f[item-1]),x)+b[item-1]

def getOptimalAssortment(Theta,b,Nx,x):
    #print(Theta)
    nx = len(Nx)
    wx = [0]*nx
    rx = [0]*nx
    for i in range(nx):
        #print(Theta[Nx[i]-1])
        wx[i] = np.exp(getUtility(Theta,b,Nx[i],x))
        rx[i] = r[Nx[i]-1]
    opt_as = Optimal_Assortment.getOptimalAssortment(n = nx, w = wx, r = rx, B=B, log = False)
    for i in range(len(opt_as)):
        opt_as[i] = Nx[opt_as[i]]
        
    return opt_as

def getProbability(ast,x):
    n = len(ast)
    wx = [0]*n
    for i in range(n):
        wx[i] = np.exp(getUtility(Theta_g,b_g,ast[i],x))
    
    wx = [1]+wx
    sum_wx = sum(wx)
    for i in range(len(wx)): 
        wx[i]/=sum_wx
    
    return wx

def getOptimalValue(ast,x):
    prob = getProbability(ast, x)
    sum = 0
    for i in range(1,len(prob)):
        sum+=r[ast[i-1]-1]*prob[i]
    return sum
        
def getCustomerPick(ast,x):
    prob = getProbability(ast, x)
    draw = np.random.choice([0]+ast,1,p=prob)
    return draw[0]



def PAO_TS_P(T,r):
    #history trajectory
    H_TS=[]
    reward = []
    reward_ora = []
    for t in range(1,T+1):
        x = Receive_x()
        Nx = Prod(x)
        if len(H_TS)==0:
            Theta_ts = np.random.normal(0,1.0,size=(D,L))
            b_ts = np.random.normal(0,1,size = len(N))
        else:
            Theta_ts,b_ts = Generate_theta_p(*zip(*H_TS),N = len(N),L = L,prod_f = prod_f)
        
        opt_as_ts = getOptimalAssortment(Theta_ts,b_ts, Nx, x)
        opt_as_ora = getOptimalAssortment(Theta_g,b_g, Nx, x)
        
        getOptimalValue(opt_as_ora, x)
        
        I_t = getCustomerPick(opt_as_ts,x)
        
        reward.append(getOptimalValue(opt_as_ts, x))
        reward_ora.append(getOptimalValue(opt_as_ora,x))
        
        H_TS.append([x,opt_as_ts,I_t])
    
    return reward,reward_ora


if __name__=='__main__':
    T = 50
    reward,reward_ora = PAO_TS_P(T,r)
    x = list(range(T))
    plt.plot(x,reward,label="reward",linestyle="-", marker="^")
    plt.plot(x,reward_ora,label="reward_ora",linestyle="-", marker="s")
    plt.show()

    res = {"reward":reward,"reward_ora":reward_ora}

    with open("test.json", 'w') as f:
        json.dump(res, f)
