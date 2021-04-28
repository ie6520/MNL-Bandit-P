
import numpy as np
import Optimal_Assortment

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

def Generate_Theta(Nx):
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
    wx = np.array([0]*n)
    for i in range(n):
        wx[i] = np.exp(np.dot(Theta_g[ast[i]:],x))
    
    sum = np.sum(wx)+1
    wx = [1]+wx
    wx/=sum
    
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
        Theta_ts = Generate_theta(Nx)
        
        opt_as_ts = getOptimalAssortment(Theta_ts, Nx, x)
        opt_as_ora = getOptimalAssortment(Theta_g, Nx, x)
        
        getOptimalValue(opt_as_ora, x)
        
        I_t = getCustomerPick(opt_as_ts,x)
        
        H_TS.append([x,opt_as_ts,I_t])
        
if __name__=='__main__':
    print('test')

    
        