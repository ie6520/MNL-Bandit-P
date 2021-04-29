import numpy as np
import matplotlib.pyplot as plt

from sample import Generate_theta
import json

with open("test.json", 'r') as f:
    data = json.load(f)
    reward = data["reward"]
    reward_ora = data["reward_ora"]

t = len(reward)
x = list(range(t))
reward = [sum(reward[:i]) for i in range(t)]
reward_ora = [sum(reward_ora[:i]) for i in range(t)]

plt.plot(x,reward,label="reward",linestyle="-")
plt.plot(x,reward_ora,label="reward_ora",linestyle="-")
plt.legend()
plt.show()