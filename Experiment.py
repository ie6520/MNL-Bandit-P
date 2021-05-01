import PAO_TS as PT 
import PAO_TS as PTP 
import numpy as np 



#feature dimension
D = 5
K = 6
#item set
N = [i+1 for i in range(20)]
#cardinality constraint
B = 20

r = np.random.uniform(size = len(N))
#2d-array
Theta_p_g=(2*np.random.normal(0,1.0,size=(D,K))-np.random.uniform(size = (D,K)))/np.sqrt(D)

prod_f = np.random.normal(0,1.0,size = (len(N),K))

Theta_np_g=np.matmul(prod_f,np.transpose(Theta_p_g)) 

def Receive_x():
    return np.random.rand(D)


print(np.shape(Theta_np_g))
print(np.shape(PT.Theta_g))

T = 5

customer_list = [Receive_x() for i in range(T)]

re1,reo1 = PT.PAO_TS(T,r,customer_list)
re2,reo2 = PT.PAO_TS_P(T,r,customer_list)

res = {"reward1":re1,"reward_ora1":reo1,"reward2":re2,"reward_ora2":reo2}

with open("test.json", 'w') as f:
    json.dump(res, f)


