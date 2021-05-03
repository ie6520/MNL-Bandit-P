import numpy as np
import matplotlib.pyplot as plt
import math as mt

# MNL model
# S: assortment 
# X: feature 
# z: feature weight

def MNL(z,S,X):
    prob = [0 for _ in range(len(S)+1)]
    cpv = [mt.exp(np.dot(z, np.transpose(X[S[i]]))) for i in range(len(S))]
    base = sum(cpv)+1
    prob[0] = 1/base
    for i in range(1,len(prob)):
        prob[i] = cpv[i-1]/base
    return np.array(prob)

# the argmax to provide assortment St
def getOptimalAssortment(r, N, z, X):
    r_sorted = r
    r_sorted.sort(reverse=True)
    N_sorted = np.argsort(r)
    N_sorted = N_sorted[::-1]
    baseline = MNL(z, [N_sorted[0]], X)[1]*r_sorted[0]
    for i in range(2,len(N)):
        potential_assortment = [N_sorted[l] for l in range(i)]
        proba = MNL(z, potential_assortment, X)
        new = sum([proba[j+1]*r_sorted[j] for j in range(i)])
        if new >= baseline:
            baseline = new
        else:
            break
    return np.array([N_sorted[m] for m in range(i-1)])

if __name__=='__main__':
    print()