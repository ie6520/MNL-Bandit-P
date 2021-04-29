
import numpy as np
import Optimal_Assortment

from sample import Generate_theta

#feasure dimension
D = 1
#item set
N = [1,2,3,4,5]
#cardinality constraint
B = 6

r = [5,4,3,2,1]
#2d-array
Theta_g=[[]]



def Receive_x():
    x = 1
    return x

def Prod(x):
    N_x = []
    return N

def getOptimalAssortment(Theta,Nx,x):
    nx = len(Nx)
    wx = [0]*nx
    rx = [0]*nx
    for i in range(nx):
        wx[i] = np.exp(np.dot(Theta[Nx[i]:],x))
        rx[i] = r[Nx[i]]
    opt_as = Optimal_Assortment.getOptimalAssortment(n = nx, w = wx, r = rx, B=B, log = True)
    for i in range(nx):
        opt_as[i] = Nx[opt_as[i]]
        
    return opt_as

def getProbability(ast,x):
    n = len(ast)
    wx = [0]*n
    for i in range(n):
        wx[i] = np.exp(np.dot(Theta_g[ast[i]:],np.array(x)))
    
    wx = [1]+wx
    sum = np.sum(wx)
    for i in range(len(wx)): 
        wx[i]/=sum
    
    return wx

def getOptimalValue(ast,x):
    prob = getProbability(ast, x)
    sum = 0
    for i in range(1,len(ast)):
        sum+=r[ast[i-1]]*prob[i]
    return sum
        
def getCustomerPick(ast,x):
    prob = getProbability(ast, x)
    draw = np.random.choice([0]+ast,1,p=prob)
    return draw[0]



def PAO_TS(T,r):
    #history trajectory
    H_TS=[]
    for t in range(1,T+1):
        x = Receive_x()
        Nx = Prod(x)
        if len(H_TS)==0:
            Theta_ts = np.random.normal(0,1,size=(len(N),D))
        else:
            Theta_ts = Generate_theta(*zip(*H_TS),len(N))
        
        opt_as_ts = getOptimalAssortment(Theta_ts, Nx, x)
        opt_as_ora = getOptimalAssortment(Theta_g, Nx, x)
        
        getOptimalValue(opt_as_ora, x)
        
        I_t = getCustomerPick(opt_as_ts,x)
        
        H_TS.append([x,opt_as_ts,I_t])
        
if __name__=='__main__':
    print('test')

    a = [1,2,3]
    
    sum = np.sum(a)+1
    
    
    print(a)
    
        