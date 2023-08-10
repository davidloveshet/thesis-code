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

budget_list = [   0.3, 0.8, 1, 3 , 8 , 10 , 20, 30, 40 , 50, 60, 70    ]  # 单位为 J，能量大小


# --------------------------------------------------------------------------------#
#   M = 1
average_reward_set_M_1_ = np.array([57.576   , 62.262   , 63.7344  , 69.5976  , 72.2754  , 72.5868  ,
       73.37664 , 73.81104 , 73.84752 , 74.047536, 74.15988 ,  74.2094      ])

optimal_average_reward_M_1 = np.ones(len(budget_list)) * 74.6514


regret_M_1 = []
for i in range(len(budget_list)):
    regret_M_1.append( budget_list[i] * ( optimal_average_reward_M_1[i] - average_reward_set_M_1_[i]) )
    
# --------------------------------------------------------------------------------#

    
# --------------------------------------------------------------------------------#  
#  M = 2 
average_reward_set_M_2_ = np.array([54.32      , 59.28      , 60.048     , 63.512     , 64.5555    ,
       65.2008    , 65.9832    , 66.4324    , 66.4512    , 66.61392   ,
       66.7584    , 66.76131429])

optimal_average_reward_M_2 = np.ones(len(budget_list)) * 67.1751

regret_M_2 = []
for i in range(len(budget_list)):
    regret_M_2.append( budget_list[i] * ( optimal_average_reward_M_2[i] - average_reward_set_M_2_[i]) )
    
# --------------------------------------------------------------------------------#    
    
# --------------------------------------------------------------------------------#    
#  M = 3   
average_reward_set_M_3_ = np.array([51.72      , 53.91      , 55.2       , 57.496     , 58.467     ,
       58.6476    , 59.4756    , 59.7868    , 59.9787    , 60.13848   ,
       60.12      , 60.25028571])

optimal_average_reward_M_3 = np.ones(len(budget_list)) * 60.65511429

regret_M_3 = []
for i in range(len(budget_list)):
    regret_M_3.append( budget_list[i] * ( optimal_average_reward_M_3[i] - average_reward_set_M_3_[i]) )
    
# --------------------------------------------------------------------------------#    
    
# --------------------------------------------------------------------------------#
#   M = 4 

average_reward_set_M_4_ = np.array([50.6   , 50.595 , 51.864 , 52.94  , 53.2485, 53.3496, 53.8074,
       54.022 , 54.1704, 54.1032, 54.2826, 54.2916])

optimal_average_reward_M_4 = np.ones(len(budget_list)) * 54.75574286  

regret_M_4 = []
for i in range(len(budget_list)):
    regret_M_4.append( budget_list[i] * ( optimal_average_reward_M_4[i] - average_reward_set_M_4_[i]) )
   
# --------------------------------------------------------------------------------#    
   
    
 
plt.plot(budget_list , average_reward_set_M_1_, '-o', label = 'M=1', LineWidth = '2')
plt.plot(budget_list , optimal_average_reward_M_1, '--', label = 'M=1,optimal', LineWidth = '2')

plt.plot(budget_list , average_reward_set_M_2_, '-^', label = 'M=2', LineWidth = '2')
plt.plot(budget_list , optimal_average_reward_M_2, '--', label = 'M=2,optimal', LineWidth = '2')

plt.plot(budget_list , average_reward_set_M_3_, '-.', label = 'M=3', LineWidth = '2')
plt.plot(budget_list , optimal_average_reward_M_3, '--', label = 'M=3,optimal', LineWidth = '2')

plt.plot(budget_list , average_reward_set_M_4_, '-*', label = 'M=4', LineWidth = '2')
plt.plot(budget_list , optimal_average_reward_M_4, '--', label = 'M=4,optimal', LineWidth = '2')

 
plt.xlabel('energy (J)',fontsize=14)
plt.ylabel('throughput/energy (M bits/J)',fontsize=14)
 
plt.legend( ncol=1, fontsize=11, framealpha = 0)
plt.grid()
plt.xlim([ -5, 112])
plt.title("Rayleigh Fading Channel")
# plt.ylim([ 35, 77 ])

fig.savefig('scenario_II_Rayleigh.eps', dpi = 600, format = 'eps')




 


# In[11] Ricean
import numpy as np
import random 
import matplotlib
import matplotlib.pyplot as plt

fig, ax = plt.subplots()


budget_list = [   0.3, 0.8, 1, 3 , 8 , 10 , 20, 30    ]  # 单位为 J，能量大小



# --------------------------------------------------------------------------------#
#   M = 1
average_reward_set_M_1_ = np.array([ 98.28  , 102.765 , 104.328 , 107.712 , 109.4265, 109.9236,
       110.1294, 110.2704])

optimal_average_reward_M_1 = np.ones(len(budget_list)) * 110.907


regret_M_1 = []
for i in range(len(budget_list)):
    regret_M_1.append( budget_list[i] * ( optimal_average_reward_M_1[i] - average_reward_set_M_1_[i]) )
    
# --------------------------------------------------------------------------------#

    
# --------------------------------------------------------------------------------#  
#  M = 2 
average_reward_set_M_2_ = np.array([89.784  , 93.231  , 94.0752 , 97.524  , 98.8056 , 99.10656,
       99.37188, 99.74616])

optimal_average_reward_M_2 = np.ones(len(budget_list)) * 100.8108

regret_M_2 = []
for i in range(len(budget_list)):
    regret_M_2.append( budget_list[i] * ( optimal_average_reward_M_2[i] - average_reward_set_M_2_[i]) )
    
# --------------------------------------------------------------------------------#    
    
# --------------------------------------------------------------------------------#    
#  M = 3   
average_reward_set_M_3_ = np.array([84.36   , 85.311  , 86.0688 , 88.8048 , 89.5995 , 89.75376,
       90.39744, 90.32088])

optimal_average_reward_M_3 = np.ones(len(budget_list)) * 90.9354

regret_M_3 = []
for i in range(len(budget_list)):
    regret_M_3.append( budget_list[i] * ( optimal_average_reward_M_3[i] - average_reward_set_M_3_[i]) )
    
# --------------------------------------------------------------------------------#    
    
# --------------------------------------------------------------------------------#
#   M = 4 

average_reward_set_M_4_ = np.array([77.496  , 79.173  , 79.3368 , 80.7744 , 81.1999 , 81.59544,
       81.99684, 82.08432])

optimal_average_reward_M_4 = np.ones(len(budget_list)) * 82.4292

regret_M_4 = []
for i in range(len(budget_list)):
    regret_M_4.append( budget_list[i] * ( optimal_average_reward_M_4[i] - average_reward_set_M_4_[i]) )
   
# --------------------------------------------------------------------------------#    
   
    
 
plt.plot(budget_list , average_reward_set_M_1_, '-o', label = 'M=1', LineWidth = '2')
plt.plot(budget_list , optimal_average_reward_M_1, '--', label = 'M=1,optimal', LineWidth = '2')

plt.plot(budget_list , average_reward_set_M_2_, '-^', label = 'M=2', LineWidth = '2')
plt.plot(budget_list , optimal_average_reward_M_2, '--', label = 'M=2,optimal', LineWidth = '2')

plt.plot(budget_list , average_reward_set_M_3_, '-.', label = 'M=3', LineWidth = '2')
plt.plot(budget_list , optimal_average_reward_M_3, '--', label = 'M=3,optimal', LineWidth = '2')

plt.plot(budget_list , average_reward_set_M_4_, '-*', label = 'M=4', LineWidth = '2')
plt.plot(budget_list , optimal_average_reward_M_4, '--', label = 'M=4,optimal', LineWidth = '2')

 
plt.xlabel('energy (J)',fontsize=14)
plt.ylabel('throughput/energy (M bits/J)',fontsize=14)
 
plt.legend( ncol=1, fontsize=11, framealpha = 0)
plt.grid()
plt.xlim([ -3, 50 ])
plt.title("Ricean Fading Channel")


fig.savefig('scenario_II_Ricean.eps', dpi = 600, format = 'eps')







