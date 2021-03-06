import numpy as np
from numpy.lib.function_base import place
import nested_Assortment
import matplotlib.pyplot as plt

from sample import Generate_theta
from sample import Generate_theta_p
import time
import json


#feature dimension
D = 1
K = 10
#item set
N = [i+1 for i in range(20)]

r =np.random.uniform(0,1,size = len(N))
r.sort()
print(r)


#2d-array
Theta_g_p=(2*np.random.normal(0,1,size=(D,K))-np.random.uniform(0,1,size=(D,K)))/np.sqrt(D*K)
prod_f = (2*np.random.normal(0,1,size=(len(N),K))+2*np.random.uniform(size=(len(N),K)))/np.sqrt(K)
print(prod_f)

def getInferredTheta(Theta_p):
    return np.matmul(prod_f,np.transpose(Theta_p))

#Theta_g_np=np.matmul(prod_f,np.transpose(Theta_p_g)) 
Theta_g_np=getInferredTheta(Theta_g_p)

print(Theta_g_np)

def Receive_x():
    return [1]

def Prod(x):
    N_x = []
    return N

def getOptimalAssortment(z,Nx,x,prod_f):
    #print(Theta)
    nx = len(Nx)
    #wx = [0]*nx
    rx = [0]*nx
    #Theta = getInferredTheta(z)
    for i in range(nx):
        #print(Theta[Nx[i]-1])
        #wx[i] = np.exp(np.dot(Theta[Nx[i]-1],x))
        rx[i] = r[Nx[i]-1]
    #opt_as = nested_Assortment.getOptimalAssortment(n = nx, w = wx, r = rx, B=B, log = False)
    opt_as = nested_Assortment.getOptimalAssortment(r=rx, N=Nx, z=z, X=prod_f)
    for i in range(len(opt_as)):
        opt_as[i] = Nx[opt_as[i]]
    return opt_as

def getProbability(Theta,ast,x):
    n = len(ast)
    wx = [0]*n
    for i in range(n):
        wx[i] = np.exp(np.dot(Theta[ast[i]-1],np.array(x)))
    
    wx = [1]+wx
    sum_wx = sum(wx)
    for i in range(len(wx)): 
        wx[i]/=sum_wx
    return wx

def getOptimalValue(Theta,ast,x):
    prob = getProbability(Theta,ast,x)
    sum = 0
    for i in range(len(ast)):
        sum+=r[ast[i]-1]*prob[i+1]
    return sum

def getCustomerPick(Theta,ast,x):
    prob = getProbability(Theta,ast,x)
    ast = ast.tolist()
    ast.insert(0,0)
    draw = np.random.choice(ast,1,p=prob)
    return draw[0]

start_time = time.time()
def PAO_TS_exp(T,r):
    #history trajectory
    H_TS_p=[]
    reward_p = []
    reward_ora = []
    for t in range(1,T+1):
        x = Receive_x()
        Nx = Prod(x)

        if len(H_TS_p)==0:
            Theta_ts_p = np.random.normal(0,1.0,size=(D,K))/np.sqrt(K)
        else:
            Theta_ts_p = Generate_theta_p(*zip(*H_TS_p),N = len(N),K = K,prod_f = prod_f)

        opt_as_ts_p = getOptimalAssortment(Theta_ts_p,Nx,x,prod_f)
        opt_as_ora = getOptimalAssortment(Theta_g_p, Nx,x,prod_f)
        #Theta_g_np dimension N*1
        I_t_p = getCustomerPick(Theta_g_np,opt_as_ts_p,x)

        reward_p.append(getOptimalValue(Theta_g_np,opt_as_ts_p,x))
        reward_ora.append(getOptimalValue(Theta_g_np,opt_as_ora,x))
        
        H_TS_p.append([x,opt_as_ts_p,I_t_p])

        print(np.max(np.abs(getInferredTheta(Theta_ts_p)-Theta_g_np).flatten()))
        print(sum(reward_p)-sum(reward_ora))
        print(opt_as_ora,opt_as_ts_p)
    
    return reward_p,reward_ora

T = 10
reward_p,reward_ora = PAO_TS_exp(T,r)

end_time= time.time()
print("The time for TS is:", end_time-start_time)

res = {"reward_p":reward_p,"reward_ora":reward_ora}

with open("test.json", 'w') as f:
    json.dump(res, f)