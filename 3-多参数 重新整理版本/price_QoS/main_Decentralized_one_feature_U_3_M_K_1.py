# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 08:41:00 2021

@author: xxx

 
"""

# test.file
import numpy as np
from numpy.linalg import pinv
import math
import matplotlib.pyplot as plt
from my_SW_LinUCB_static import my_SW_LinUCB_static, my_SW_LinUCB_comparison
# from feature_generation import feature_generation
# from theta_generation import theta_generation
import copy
# from softmax_standard import Softmax
from hungarian import Hungarian
from random import shuffle


dimension = 2
feature = np.zeros(dimension).reshape(dimension,1)
lambdada = 1
alpha = 1 # the algorithm parameter
transmission_duration = 1e-03 # ms 
ALGO = 'random' # 'random', 'fixed', 'proposed'


# In[11] 
U = 3   # there are three SUs
M_U = 1 # channels that one SU senses
K_U = 1 # channels that one SU accesses
K = U
# In[1] define the channel set for SU 1

arm_theta_set_SU_1 = [
np.array([0.85, -0.91]),
np.array([0.80, -0.43]),
np.array([0.75, -0.33]),
np.array([0.53, -0.45]),
np.array([0.23, -0.41]),
np.array([0.43, -0.55]),
np.array([0.48, -0.31]),
np.array([0.59, -0.53]),
np.array([0.53, -0.45])
]

arm_theta_set_SU_1 = [ arm_theta_set_SU_1[i].reshape(dimension,1) for i in range(9) ]
 
arm_theta_set = arm_theta_set_SU_1
theta_generation_1 = arm_theta_set_SU_1
Channel_Num = len(arm_theta_set)
 
# In[12]
T_simulation = 16000 # the overall simulation time
m_c_number = 15  # Monte Carlo Number 
 
Reward_all_simulation = np.zeros(T_simulation)
Reward_optimal = np.zeros(T_simulation)

T_init = 1           # the initial verification time
# In[0]

TRAFFIC_TYPE = '2' # '1', '2' select various values, we have various QoS requirements 


if TRAFFIC_TYPE == '1':
    feature_generation = np.array([  [1], [0]  ])
elif TRAFFIC_TYPE == '2':
    feature_generation = np.array([  [0.9701], [0.2425]  ])
  
# there exist U SUs, where U = 3
Channel_value_order = []
for i in range(len(arm_theta_set)):
    Channel_value_order.append(np.dot(feature_generation.T, arm_theta_set_SU_1[i]))
    
# In[11] Benchmark 

Reward_optimal_mc = np.ones((m_c_number, T_simulation)) * sum(sorted(Channel_value_order, reverse = True)[0:K])
 
# In[2] generate the policy set
policy_SU_1 = []
policy_SU_2 = []
policy_SU_3 = []
policy = [ policy_SU_1, policy_SU_2, policy_SU_3 ]

for k, arm_theta in enumerate(arm_theta_set):
    # policy.append(my_SW_LinUCB_static(arm_theta, feature, dimension, lambdada))
    policy_SU_1.append(my_SW_LinUCB_comparison([], feature, dimension, lambdada))
    policy_SU_2.append(my_SW_LinUCB_comparison([], feature, dimension, lambdada))
    policy_SU_3.append(my_SW_LinUCB_comparison([], feature, dimension, lambdada))
 
# In[6] Simulate more channels at one slot, there are N channels, M channels each slot

Reward = []

Reward_all_simulation_mc = np.zeros((m_c_number, T_simulation))
Reward_all_simulation_mc_SU_1 = np.zeros((m_c_number, T_simulation))
Reward_all_simulation_mc_SU_2 = np.zeros((m_c_number, T_simulation))
Reward_all_simulation_mc_SU_3 = np.zeros((m_c_number, T_simulation))

reward_difference_mc = np.zeros((m_c_number, T_simulation))

collision_Num = np.zeros((m_c_number, T_simulation))
Throughput_mc_1 = np.zeros((m_c_number, T_simulation))
Price_cost_mc_1 = np.zeros((m_c_number, T_simulation))

Throughput_mc_2 = np.zeros((m_c_number, T_simulation))
Price_cost_mc_2 = np.zeros((m_c_number, T_simulation))

Throughput_mc_3 = np.zeros((m_c_number, T_simulation))
Price_cost_mc_3 = np.zeros((m_c_number, T_simulation))

len_Num = len(arm_theta_set_SU_1)

for m_c in range(m_c_number):
    
    print("This is the", m_c, "simulation")
    for i in range(Channel_Num):
        policy_SU_1[i].re_init()
        policy_SU_2[i].re_init()
        policy_SU_3[i].re_init()
        
    reward_SU_1 = np.zeros(Channel_Num)
    reward_SU_2 = np.zeros(Channel_Num)
    reward_SU_3 = np.zeros(Channel_Num)
    channel_state_1 = np.zeros(Channel_Num)
    channel_state_2 = np.zeros(Channel_Num)
    channel_state_3 = np.zeros(Channel_Num)
    
    hat_theta_one = np.zeros((dimension,1)).reshape(dimension,1) 
    hat_theta_SU_1 = []
    hat_theta_SU_2 = []
    hat_theta_SU_3 = []
    for i in range(Channel_Num):
        hat_theta_SU_1.append(hat_theta_one)
        hat_theta_SU_2.append(hat_theta_one)
        hat_theta_SU_3.append(hat_theta_one)
    ucb_s_SU_1 = np.zeros(Channel_Num)
    ucb_s_SU_2 = np.zeros(Channel_Num)
    ucb_s_SU_3 = np.zeros(Channel_Num)
    
    for t in range(T_simulation):
    
        
        # '''
        # Our proposed method
        if t < T_init: # initialization, each channel has been sensed
            for i, theta_index in enumerate(arm_theta_set):
                channel_state_1[i] = np.random.binomial(1,theta_generation_1[i][0],1)
                reward_SU_1[i] = feature_generation[0]*channel_state_1[i] + feature_generation[1]*theta_generation_1[i][1]  
                hat_theta_SU_1[i] = policy_SU_1[i].update_information(feature_generation, reward_SU_1[i])
                ucb_s_SU_1[i] = policy_SU_1[i].compute_index(hat_theta_SU_1[i], feature_generation, policy_SU_1[i].invcov, t+1)
 
                hat_theta_SU_2[i] = policy_SU_2[i].update_information(feature_generation, reward_SU_1[i])
                ucb_s_SU_2[i] = policy_SU_2[i].compute_index(hat_theta_SU_2[i], feature_generation, policy_SU_2[i].invcov, t+1)                

                hat_theta_SU_3[i] = policy_SU_3[i].update_information(feature_generation, reward_SU_1[i])
                ucb_s_SU_3[i] = policy_SU_3[i].compute_index(hat_theta_SU_3[i], feature_generation, policy_SU_3[i].invcov, t+1)   
        
        else:
            for i in range(Channel_Num):
                channel_state_1[i] = np.random.binomial(1,theta_generation_1[i][0],1)
                channel_state_2[i] = np.random.binomial(1,theta_generation_1[i][0],1)
                channel_state_3[i] = np.random.binomial(1,theta_generation_1[i][0],1)
                reward_SU_1[i] = feature_generation[0]*channel_state_1[i] + feature_generation[1]*theta_generation_1[i][1]  
                reward_SU_2[i] = feature_generation[0]*channel_state_2[i] + feature_generation[1]*theta_generation_1[i][1]  
                reward_SU_3[i] = feature_generation[0]*channel_state_3[i] + feature_generation[1]*theta_generation_1[i][1]  
            # then SUs start to access channels, where SU 1 accesses channel 1, SU 2 accesses channel 2...
            mixer = np.random.random(ucb_s_SU_1.size)
            ucb_indices = list(np.lexsort((mixer, ucb_s_SU_1))) 
            
            # '''
            if ALGO == 'random':
                shuffle(ucb_indices)
            elif ALGO == 'fixed':
                ucb_indices = [8,7,6]
            # '''
            output = ucb_indices[::-1]
            channel_access_1 = output[ (1+t) % U ]  # M channels are chosen to be sensed 
            hat_theta_SU_1[channel_access_1] = policy_SU_1[channel_access_1].update_information(feature_generation, reward_SU_1[channel_access_1])
            policy_SU_1[channel_access_1].count_the_sensed_time()
            if channel_state_1[channel_access_1] == 1:
                Throughput_mc_1[m_c][t] += channel_state_1[channel_access_1] # theta_generation_1[channel_access_1][0]
                Price_cost_mc_1[m_c][t] += theta_generation_1[channel_access_1][1]
            Reward_all_simulation_mc_SU_1[m_c][t] = np.dot(feature_generation.T, theta_generation_1[channel_access_1])

            mixer = np.random.random(ucb_s_SU_2.size)
            ucb_indices = list(np.lexsort((mixer, ucb_s_SU_2))) 
            
            # '''
            if ALGO == 'random':
                shuffle(ucb_indices)
            elif ALGO == 'fixed':
                ucb_indices = [8,7,6]
            # '''
            output = ucb_indices[::-1]            
            channel_access_2 = output[ (2+t) % U ]
            hat_theta_SU_2[channel_access_2] = policy_SU_2[channel_access_2].update_information(feature_generation, reward_SU_2[channel_access_2])
            policy_SU_2[channel_access_2].count_the_sensed_time()
            if channel_state_2[channel_access_2] == 1:
                Throughput_mc_2[m_c][t] += channel_state_2[channel_access_2] # theta_generation_1[channel_access_2][0]
                Price_cost_mc_2[m_c][t] += theta_generation_1[channel_access_2][1]
            Reward_all_simulation_mc_SU_2[m_c][t] = np.dot(feature_generation.T, theta_generation_1[channel_access_2])

            mixer = np.random.random(ucb_s_SU_3.size)
            ucb_indices = list(np.lexsort((mixer, ucb_s_SU_3)))
            
            # '''
            if ALGO == 'random':
                shuffle(ucb_indices)
            elif ALGO == 'fixed':
                ucb_indices = [8,7,6]
            # '''
            output = ucb_indices[::-1]               
            channel_access_3 = output[ (3+t) % U ]
            hat_theta_SU_3[channel_access_3] = policy_SU_3[channel_access_3].update_information(feature_generation, reward_SU_3[channel_access_3])
            policy_SU_3[channel_access_3].count_the_sensed_time()   
            if channel_state_3[channel_access_3] == 1:
                Throughput_mc_3[m_c][t] += channel_state_3[channel_access_3] # theta_generation_1[channel_access_3][0]
                Price_cost_mc_3[m_c][t] += theta_generation_1[channel_access_3][1]                  
            Reward_all_simulation_mc_SU_3[m_c][t] = np.dot(feature_generation.T, theta_generation_1[channel_access_3])
            
            # if t % 1000 == 0:
            #     print(channel_access_1, channel_access_2, channel_access_3)
            
            if channel_access_1 == channel_access_2 :
                Reward_all_simulation_mc_SU_1[m_c][t] = 0
                Reward_all_simulation_mc_SU_2[m_c][t] = 0
                collision_Num[m_c][t] += 1
                Throughput_mc_1[m_c][t] = 0
                Throughput_mc_2[m_c][t] = 0
                
            elif channel_access_1 == channel_access_3:
                Reward_all_simulation_mc_SU_1[m_c][t] = 0
                Reward_all_simulation_mc_SU_3[m_c][t] = 0
                collision_Num[m_c][t] += 1
                Throughput_mc_1[m_c][t] = 0
                Throughput_mc_3[m_c][t] = 0
                
            elif channel_access_2 == channel_access_3:
                Reward_all_simulation_mc_SU_2[m_c][t] = 0
                Reward_all_simulation_mc_SU_3[m_c][t] = 0
                collision_Num[m_c][t] += 1
                Throughput_mc_2[m_c][t] = 0
                Throughput_mc_3[m_c][t] = 0
                
            for element in range(Channel_Num):    
                ucb_s_SU_1[element] = policy_SU_1[element].compute_index(hat_theta_SU_1[element], feature_generation, policy_SU_1[element].invcov, t+1)
                ucb_s_SU_2[element] = policy_SU_2[element].compute_index(hat_theta_SU_2[element], feature_generation, policy_SU_2[element].invcov, t+1)
                ucb_s_SU_3[element] = policy_SU_3[element].compute_index(hat_theta_SU_3[element], feature_generation, policy_SU_3[element].invcov, t+1)
                
            if t % 1000 == 0:
                print("The time is", t )
   
  
# In[8] plot the figure for regret 

  
t_x = list(range(T_simulation))
Reward_all_simulation_mc = Reward_all_simulation_mc_SU_1 + Reward_all_simulation_mc_SU_2 + Reward_all_simulation_mc_SU_3
reward_difference_mc = Reward_optimal_mc - Reward_all_simulation_mc

regret_x_label = [ round(0.005 * T_simulation), round(0.01 * T_simulation), round(0.03 * T_simulation), round(0.05 * T_simulation),  round(0.07 * T_simulation), round(0.1 * T_simulation), round(0.2 * T_simulation), round(0.3 *T_simulation), round(0.4 *T_simulation), round(0.5 *T_simulation), round(0.6 *T_simulation), round(0.7 *T_simulation), round(0.8 *T_simulation), round(0.9 *T_simulation),  T_simulation  ]


cumulative_regret_y_label   = np.zeros(len(regret_x_label))
accumulative_reward_y_label = np.zeros(len(regret_x_label))
cumulative_collision_y_label   = np.zeros(len(regret_x_label))

for i,element in enumerate(regret_x_label):
    for j in range(reward_difference_mc.shape[0]):
        cumulative_regret_y_label[i] += sum(reward_difference_mc[j,0:element])
        accumulative_reward_y_label[i] += sum(Reward_all_simulation_mc[j][0:element])
        cumulative_collision_y_label[i] += sum(collision_Num[j][0:element])/m_c_number
    cumulative_regret_y_label[i] = cumulative_regret_y_label[i]/(m_c_number  )
    accumulative_reward_y_label[i] = accumulative_reward_y_label[i]/(m_c_number*element)
 
    
plt.plot(regret_x_label, cumulative_regret_y_label,'b-^')
# plt.plot(regret_x_label, accumulative_reward_y_label)
plt.show()
 

# In[1] chosen time

chosen_time_matrix_1 = []
chosen_time_matrix_2 = []
chosen_time_matrix_3 = []
for n in range(len_Num):
    chosen_time_matrix_1.append( policy_SU_1[n].t_count )
    chosen_time_matrix_2.append( policy_SU_2[n].t_count )
    chosen_time_matrix_3.append( policy_SU_3[n].t_count )


# In[2] price 

Price_1 = -sum(Price_cost_mc_1)/m_c_number
Price_all_1 = sum(Price_1)

Price_2 = -sum(Price_cost_mc_2)/m_c_number
Price_all_2 = sum(Price_2)

Price_3 = -sum(Price_cost_mc_3)/m_c_number
Price_all_3 = sum(Price_3)


cumulative_Price_y_label_1   = np.zeros(len(regret_x_label))
cumulative_Price_y_label_2   = np.zeros(len(regret_x_label))
cumulative_Price_y_label_3   = np.zeros(len(regret_x_label))
average_Price_1 = np.zeros(len(regret_x_label))
average_Price_2 = np.zeros(len(regret_x_label))
average_Price_3 = np.zeros(len(regret_x_label))


for i,element in enumerate(regret_x_label):
    cumulative_Price_y_label_1[i] += sum(Price_1[0:element])
    average_Price_1[i] = cumulative_Price_y_label_1[i]/element
    
    cumulative_Price_y_label_2[i] += sum(Price_2[0:element])
    average_Price_2[i] = cumulative_Price_y_label_2[i]/element

    cumulative_Price_y_label_3[i] += sum(Price_3[0:element])
    average_Price_3[i] = cumulative_Price_y_label_3[i]/element
    
    # accumulative_reward_y_label[i] += sum(Reward_all_simulation_mc[j][0:element])
    # cumulative_regret_y_label[i] = cumulative_regret_y_label[i]/(m_c_number  )
    # accumulative_reward_y_label[i] = accumulative_reward_y_label[i]/( element)

# plt.plot(regret_x_label, cumulative_Price_y_label_1,'b-^')
# plt.plot(regret_x_label, cumulative_Price_y_label_2,'b-^')
# plt.plot(regret_x_label, cumulative_Price_y_label_3,'b-^')
# plt.show()

plt.plot(regret_x_label, average_Price_1,'b-^')
plt.plot(regret_x_label, average_Price_2,'b-^')
plt.plot(regret_x_label, average_Price_3,'b-^')
plt.title('price')
plt.show()


# In[3] throughput 


Throuhgput_1 = sum(Throughput_mc_1)/m_c_number
Throuhgput_all_1 = sum(Throuhgput_1)


Throuhgput_2 = sum(Throughput_mc_2)/m_c_number
Throuhgput_all_2 = sum(Throuhgput_2)

Throuhgput_3 = sum(Throughput_mc_3)/m_c_number
Throuhgput_all_3 = sum(Throuhgput_3)


cumulative_throuhgput_y_label_1   = np.zeros(len(regret_x_label))
cumulative_throuhgput_y_label_2   = np.zeros(len(regret_x_label))
cumulative_throuhgput_y_label_3   = np.zeros(len(regret_x_label))
average_throughput_1 = np.zeros(len(regret_x_label))
average_throughput_2 = np.zeros(len(regret_x_label))
average_throughput_3 = np.zeros(len(regret_x_label))

for i,element in enumerate(regret_x_label):
    cumulative_throuhgput_y_label_1[i] += sum(Throuhgput_1[0:element])
    average_throughput_1[i] = cumulative_throuhgput_y_label_1[i]/element
    
    cumulative_throuhgput_y_label_2[i] += sum(Throuhgput_2[0:element])
    average_throughput_2[i] = cumulative_throuhgput_y_label_2[i]/element

    cumulative_throuhgput_y_label_3[i] += sum(Throuhgput_3[0:element])
    average_throughput_3[i] = cumulative_throuhgput_y_label_3[i]/element
    
    # accumulative_reward_y_label[i] += sum(Reward_all_simulation_mc[j][0:element])
    # cumulative_regret_y_label[i] = cumulative_regret_y_label[i]/(m_c_number  )
    # accumulative_reward_y_label[i] = accumulative_reward_y_label[i]/( element)

# plt.plot(regret_x_label, cumulative_throuhgput_y_label_1,'b-^')
# plt.plot(regret_x_label, cumulative_throuhgput_y_label_2,'b-^')
# plt.plot(regret_x_label, cumulative_throuhgput_y_label_3,'b-^')
# plt.show()

plt.plot(regret_x_label, average_throughput_1,'b-^')
plt.plot(regret_x_label, average_throughput_2,'b-^')
plt.plot(regret_x_label, average_throughput_3,'b-^')
plt.title('average throughput')
plt.show()

 

# In[4] we also show the collision 

'''
Note that we only choose one simulatio result here, so the resutls are different. But they have the identical trend 
'''
 
fig, ax = plt.subplots()
t_x_bar = list(range(len(regret_x_label)))
# plt.bar(t_x_bar, cumulative_collision_y_label, color='tab:red')
plt.bar(t_x_bar, cumulative_collision_y_label)
ax = plt.gca()  # X-Y-Z
tt = ['80', '160', '480', '800', '1120', '1600', '3200', '4800', '6400', '8000', '9600', '11200', '12800', '14400', '16000']
x = np.arange(len(regret_x_label))   # x-data number 

ax.set_xticks(x)  # x label length 
ax.set_xticklabels(tt, rotation = 45)  
# ax.set_yticks([0, 50, 100, 150, 200, 250, 300])  
ax.set_xlabel("time slot")   # x-axis name 
ax.set_ylabel("The collision number")    

# fig.savefig('s_1_M_1.eps', dpi = 600, format = 'eps')
# fig.savefig('s_2_M_1.eps', dpi = 600, format = 'eps')
 
# we record the 'cumulative_regret_y_label', 'average_throughput_1', and 'average_Price_1' in 'plot_algorithm_price_QoS_Decentral.py' file 














