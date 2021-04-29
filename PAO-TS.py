
import numpy as np
import Optimal_Assortment

from sample import Generate_theta

#feasure dimension
D = 1
#item set
N = []
#cardinality constraint
B = 6

#2d-array
Theta_g=[[]]

#history trajectory
H_TS=[]

def Receive_x():
    x = 1
    return x

def Generate_Theta(H):
    theta = 1
    return theta

def Prod(x):
    N_x = []
    return N_x

def getOptimalAssortment(Theta,Nx,x):
    nx = len(Nx)
    wx = [0]*n
    rx = [0]*n
    for i in range(n):
        wx[i] = np.exp(np.dot(Theta_ts[Nx[i]:],x))
        rx[i] = r[Nx[i]]
    opt_as = Optimal_Assortment.getOptimalAssortment(n = nx, w = wx, r = rx, B=B, log = True)
    for i in range(n):
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
    for i in range(len(ast)):
        sum+=r[ast[i]]*prob[i]
    return sum
        
def getCustomerPick(ast,x):
    prob = getProbability(ast, x)
    draw = np.random.choice([0]+ast,1,p=prob)
    return draw[0]



def PAO_TS(T,r):
    global H_TS
    
    for t in range(1,T+1):
        x = Receive_x()
        Nx = Prod(x)
        Theta_ts = Generate_theta(*zip(*H_ts),len(N))
        
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
    
        