# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 20:51:34 2021

@author: xxx
"""

# In[0] average reward Rayleigh
import numpy as np
import random 
import matplotlib
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

alpha_0 = 1/100
alpha_1 = 1/2
M = 1 


budget_list = [   0.3, 0.8, 1, 3 , 8 , 10 , 20, 30, 40 , 50    ]  # 单位为 J，能量大小


optimal_average_reward_M_1 = np.ones(len(budget_list)) * 74.6514


average_reward_set_proposed = np.array([58.56      , 63.42428571, 64.34742857, 69.336     , 72.06428571,
       72.37954286, 73.33174286, 73.81628571, 73.88254286, 74.07977143    ])

average_reward_set_proposed_epsilon_01 = np.array([60.12   , 62.055  , 62.244  , 67.764  , 69.633  , 69.2748 ,
       69.6564 , 69.9384 , 69.9957 , 69.74424])



# 不考虑能量消耗的经典 MAB 算法
average_reward_set_MAB_without_energy = np.array([36.36  , 41.985 , 40.932 , 46.224 , 51.1605, 53.8092, 59.0094,
       61.434 , 62.7876, 64.2384])

# random algorithm
average_reward_set_random = np.array([33.96  , 35.01  , 34.74  , 34.368 , 34.515 , 34.6608, 34.8894,
       34.4076, 34.6968, 34.596 ])

# c-UCB
average_reward_set_c_UCB = np.array([37.92   , 45.09   , 49.5    , 53.592  , 62.4105 , 63.8604 ,
       67.1292 , 69.4008 , 70.2945 , 71.07912])


# Budget-Limited epsilon-First Algorithm
average_reward_set_budget_limited_epsilon_First = np.array([51.96  , 52.83  , 55.224 , 51.768 , 52.506 , 54.6264, 52.1676,
       54.45  , 51.7653, 50.7132])

# EABS-UCB
average_reward_set_budget_EABS_UCB = np.array([39.12   , 37.935  , 37.944  , 40.44   , 42.831  , 43.6356 ,
       47.331  , 50.0028 , 51.6402 , 53.32896])


# EABS-TS
average_reward_set_budget_EABS_TS = np.array([45.72   , 51.93   , 50.4    , 64.884  , 70.5915 , 71.0352 ,
       72.6444 , 73.4388 , 73.6785 , 73.94832])


# ONES
average_reward_set_budget_ONES = np.array([39.     , 42.615  , 42.984  , 48.276  , 56.3985 , 58.4136 ,
       62.9892 , 66.108  , 67.6458 , 68.26752])

 

regret_M_1 = []
for i in range(len(budget_list)):
    regret_M_1.append( budget_list[i] * ( optimal_average_reward_M_1[i] - average_reward_set_proposed[i]) )
    
# --------------------------------------------------------------------------------#

    
 
   
# --------------------------------------------------------------------------------#    
   
plt.plot(budget_list , optimal_average_reward_M_1, '-', label = 'optimal', LineWidth = '2')   
 
plt.plot(budget_list , average_reward_set_proposed, '--*', label = '$\pi-$EC', LineWidth = '2')
plt.plot(budget_list , average_reward_set_MAB_without_energy, '--o', label = 'UCB', LineWidth = '2')
plt.plot(budget_list , average_reward_set_random, '--.', label = 'random algorithm', LineWidth = '2')
plt.plot(budget_list , average_reward_set_proposed_epsilon_01, '--^', label = '$\pi-$EC with $\epsilon$-greedy', LineWidth = '2')
plt.plot(budget_list , average_reward_set_c_UCB, '-^', label = 'c-UCB', LineWidth = '2')
plt.plot(budget_list , average_reward_set_budget_limited_epsilon_First, '--', label = 'Budget-Limited $\epsilon$-First', LineWidth = '2')
plt.plot(budget_list , average_reward_set_budget_EABS_UCB, '--x', label = 'EABS-UCB', LineWidth = '2')
plt.plot(budget_list , average_reward_set_budget_EABS_TS, '-o', label = 'EABS-TS', LineWidth = '2')
plt.plot(budget_list , average_reward_set_budget_ONES, '-*', label = 'ONES', LineWidth = '2')

 
plt.xlabel('energy (J)',fontsize=14)
plt.ylabel('throughput/energy (M bits/J)',fontsize=14)
 
plt.legend( ncol=1, fontsize=10, framealpha = 0)
plt.xlim([-3, 98])
plt.grid()
 
# fig.savefig('algorithm_comparison.eps', dpi = 600, format = 'eps')




 
 







