import numpy as np
import matplotlib.pyplot as plt

#from sample import Generate_theta
import json

with open("test.json", 'r') as f:
    data = json.load(f)
    reward_p = data["reward_p"]
    reward_ora = data["reward_ora"]

with open("test1.json",'r') as f:
    data = json.load(f)
    reward_algo = data["reward_p_algo"]
    reward_ora_algo = data["reward_ora_algo"]

t = len(reward_algo)
x = list(range(t))

regret_p = [sum(reward_ora[:i+1])-sum(reward_p[:i+1]) for i in range(t)]
regret_algo = [sum(reward_ora_algo[:i+1])-sum(reward_algo[:i+1]) for i in range(t)]
#plt.plot(x,ratio,label="ratio",linestyle="-")

plt.plot(x,regret_p,label="regret_TS",linestyle="-", color="r",linewidth=1.5)
plt.plot(x,regret_algo,label="regret_algo",linestyle="-", color="b",linewidth=1.5)
plt.xlabel("Number of Rounds")
plt.ylabel("Cumulative Regret")
plt.legend()
plt.show()
plt.savefig("result.png")