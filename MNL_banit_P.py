import numpy as np
import matplotlib.pyplot as plt
import math as mt
import time
import json

# MNL model
def MNL(z, S):
    global X
    prob = [0 for _ in range(len(S)+1)]
    cpv = [mt.exp(np.dot(X[S[i]],z)) for i in range(len(S))]
    base = sum(cpv)+1
    prob[0] = 1/base
    for i in range(1,len(prob)):
        prob[i] = cpv[i-1]/base
    return np.array(prob)

# the argmax to provide assortment St
def assortment_provider(z):
    global r_sorted, N_sorted
    baseline = MNL(z, [N_sorted[0]])[1]*r_sorted[0]
    for i in range(2,len(N)):
        potential_assortment = [N_sorted[l] for l in range(i)]
        proba = MNL(z, potential_assortment)
        new = sum([proba[j+1]*r_sorted[j] for j in range(i)])
        if new >= baseline:
            baseline = new
        else:
            break
    return np.array([N_sorted[m] for m in range(i-1)])

# Environment setup

start_time = time.time()

productcount = 150
productfeaturecount = 10
product_selected = ''
assortment_offered = ''
T = 200
revenue_per_round = np.empty([10,T])
revenue_per_round_optimal = np.empty([10,T])
R = 0
R_optimal = 0
N = list(range(productcount)) # set of all product

# do experiment for 10 times and get average for revenue of per round for algo and optimal 
for l in range(10):
    r = np.array([np.random.random() for _ in range(productcount)]) # revenue of products
    r_sorted = r.tolist()
    r_sorted.sort(reverse=True)
    N_sorted = np.argsort(r)
    N_sorted = N_sorted[::-1]

    # Initialize the product feature matrix
    X = (2*np.random.normal(0,1.0,size=(productcount,productfeaturecount))+2*np.random.uniform(size=(productcount,productfeaturecount)))/np.sqrt(productfeaturecount)
    #X = np.random.uniform(0.5,1,size=(productcount,productfeaturecount))
    
    # Model parameters setup
    true_beta = (2*np.random.normal(0,1.0,size=(productfeaturecount))-np.random.uniform(size=(productfeaturecount)))/np.sqrt(productfeaturecount)
    #true_beta = np.array(([(np.random.random()-0.5)*10 for _ in range(productfeaturecount)]))
    beta = [0 for _ in range(productfeaturecount)]
    Nk = np.zeros(productfeaturecount)

    # Online dynamics
    for t in range(T):
    
        # assortment by algorithm
        total = 0
        cumu = []
        if t == 0 or product_selected!='': # Exploitation
            St = assortment_provider(beta)
            assortment_offered = St
        else: # Exploration
            St = np.setdiff1d(N,assortment_offered)
            assortment_offered = St
        prob = MNL(true_beta, St)
        for i in range(len(prob)):
            total += prob[i]
            cumu.append(total)
        rand = np.random.random()
        for i in range(len(cumu)):
            if rand <= cumu[i]:
                if i == 0: # outside option chosen
                    product_t = ''
                    product_selected = ''
                    for k in St:
                        for j in range(len(Nk)):
                            #if X[k, j] == 1:
                            #    Nk[j] -= 1
                            Nk[j] -= X[k,j]
                    for j in range(len(beta)):
                        if Nk[j]>0:
                            beta[j] = np.log(Nk[j]*np.exp(1))
                        elif Nk[j]==0:
                            beta[j] = 0
                        else:
                            beta[j] = -np.log(-Nk[j]*np.exp(1))
                    break
                else: # product_t chosen
                    product_t = St[i-1]
                    for j in range(len(Nk)):
                        #if X[product_t,j] == 1:
                        #    Nk[j] += 1
                        Nk[j] += X[product_t,j]
                    for j in range(len(beta)):
                        if Nk[j]>0:
                            beta[j] = np.log(Nk[j]*np.exp(1))
                        elif Nk[j]==0:
                            beta[j] = 0
                        else:
                            beta[j] = -np.log(-Nk[j]*np.exp(1))
                    product_selected = St[i - 1]
                    revenue_per_round[l][t] = r[product_t]
                    break
    
        # optimal assortment for oracle
        total = 0
        cumu = []
        St_optimal = assortment_provider(true_beta)
        prob_optimal = MNL(true_beta, St_optimal)
        for i in range(len(prob_optimal)):
            total += prob_optimal[i]
            cumu.append(total)
        rand = np.random.random()
        for i in range(len(cumu)):
            if rand <= cumu[i]:
                if i == 0: # outside option chosen
                    product_t_optimal = ''
                    break
                else: # product_t chosen
                    product_t_optimal = St_optimal[i-1]
                    revenue_per_round_optimal[l][t] = r[product_t_optimal]
                    break
                
ave_per_round = revenue_per_round.sum(axis=0)/10
ave_per_round = ave_per_round.tolist()
ave_per_round_optimal = revenue_per_round_optimal.sum(axis=0)/10
ave_per_round_optimal= ave_per_round_optimal.tolist()

end_time = time.time()
print("The time is:", end_time-start_time)

res_algo = {"reward_p_algo":ave_per_round,"reward_ora_algo":ave_per_round_optimal}

with open("10_150_MNL.json", 'w') as f:
    json.dump(res_algo, f)