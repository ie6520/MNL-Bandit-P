from PAO-TS import PAO_TS
from PAO-TS import PAO_TS_P

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