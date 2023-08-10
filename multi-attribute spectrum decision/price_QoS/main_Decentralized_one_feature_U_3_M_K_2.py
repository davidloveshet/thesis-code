# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 08:41:00 2021

@author: xxx

TRAFFIC_TYPE = '1', '2'. Choose different value of this variable, we can obtain various QoS requirements

 
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

# In[11] 
U = 3   # there are three SUs
M_U = 2 # channels that one SU senses
K_U = 2 # channels that one SU accesses
K = 6

ALGO = 'random' # 'random', 'fixed', 'proposed'



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
m_c_number = 3

T_init = 1           # the initial verification time
Reward_all_simulation = np.zeros(T_simulation)
Reward_optimal = np.zeros(T_simulation)


# In[0]
 
TRAFFIC_TYPE = '2' # '2' 

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
 
for m_c in range(m_c_number):
    
    print("This is the", m_c, "simulation")
    for i in range(Channel_Num):
        policy_SU_1[i].re_init()
        policy_SU_2[i].re_init()
        policy_SU_3[i].re_init()
        
    reward_SU_1 = np.zeros(Channel_Num)
    reward_SU_2 = np.zeros(Channel_Num)
    reward_SU_3 = np.zeros(Channel_Num)
    channel_state = np.zeros(Channel_Num)
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
                channel_state[i] = np.random.binomial(1,theta_generation_1[i][0],1)
                reward_SU_1[i] = feature_generation[0]*channel_state[i] + feature_generation[1]*theta_generation_1[i][1]  
                hat_theta_SU_1[i] = policy_SU_1[i].update_information(feature_generation, reward_SU_1[i])
                ucb_s_SU_1[i] = policy_SU_1[i].compute_index(hat_theta_SU_1[i], feature_generation, policy_SU_1[i].invcov, t+1)
                 
                hat_theta_SU_2[i] = policy_SU_2[i].update_information(feature_generation, reward_SU_1[i])
                ucb_s_SU_2[i] = policy_SU_2[i].compute_index(hat_theta_SU_2[i], feature_generation, policy_SU_2[i].invcov, t+1)                

                hat_theta_SU_3[i] = policy_SU_3[i].update_information(feature_generation, reward_SU_1[i])
                ucb_s_SU_3[i] = policy_SU_3[i].compute_index(hat_theta_SU_3[i], feature_generation, policy_SU_3[i].invcov, t+1)   
        else:
                        
            for i in range(Channel_Num):
                channel_state[i] = np.random.binomial(1,theta_generation_1[i][0],1)
                channel_state_2[i] = np.random.binomial(1,theta_generation_1[i][0],1)
                channel_state_3[i] = np.random.binomial(1,theta_generation_1[i][0],1)
                reward_SU_1[i] = feature_generation[0]*channel_state[i] + feature_generation[1]*theta_generation_1[i][1]  
                reward_SU_2[i] = feature_generation[0]*channel_state_2[i] + feature_generation[1]*theta_generation_1[i][1] 
                reward_SU_3[i] = feature_generation[0]*channel_state_3[i] + feature_generation[1]*theta_generation_1[i][1]  
            
            
            # then SUs start to access channels

            mixer = np.random.random(ucb_s_SU_1.size)
            ucb_indices_1 = list(np.lexsort((mixer, ucb_s_SU_1))) 
            # '''
            if ALGO == 'random':
                shuffle(ucb_indices_1)
            if ALGO == 'fixed':
                ucb_indices_1 = [8,7,6,5,4,3]
            # '''
            output_1 = ucb_indices_1[::-1]
            channel_access_1 = output_1[ 2 * ((1+t) % U) : 2 * ((1+t) % U) + 2 ]  # M channels are chosen to be sensed 
            for j in range(len(channel_access_1)):
                hat_theta_SU_1[channel_access_1[j]] = policy_SU_1[channel_access_1[j]].update_information(feature_generation, reward_SU_1[channel_access_1[j]])
                policy_SU_1[channel_access_1[j]].count_the_sensed_time()
                Reward_all_simulation_mc_SU_1[m_c][t] += np.dot(feature_generation.T, theta_generation_1[channel_access_1[j]])
                if channel_state[channel_access_1[j]] == 1:
                    Throughput_mc_1[m_c][t] += channel_state[channel_access_1[j]] # theta_generation_1[channel_access_1[j]][0]
                    Price_cost_mc_1[m_c][t] += theta_generation_1[channel_access_1[j]][1]       
                
            mixer = np.random.random(ucb_s_SU_2.size)
            ucb_indices_2 = list(np.lexsort((mixer, ucb_s_SU_2)))
            # '''
            if ALGO == 'random':
                shuffle(ucb_indices_2)
            if ALGO == 'fixed':
                ucb_indices_2 = [8,7,6,5,4,3]
            # '''
            output_2 = ucb_indices_2[::-1]            
            channel_access_2 = output_2[ 2 * ((2+t) % U) : 2 * ((2+t) % U) + 2 ]
            for j in range(len(channel_access_2)):
                hat_theta_SU_2[channel_access_2[j]] = policy_SU_2[channel_access_2[j]].update_information(feature_generation, reward_SU_2[channel_access_2[j]])
                policy_SU_2[channel_access_2[j]].count_the_sensed_time()
                Reward_all_simulation_mc_SU_2[m_c][t] += np.dot(feature_generation.T, theta_generation_1[channel_access_2[j]])
                if channel_state_2[channel_access_2[j]] == 1:
                    Throughput_mc_2[m_c][t] += channel_state_2[channel_access_2[j]] # theta_generation_1[channel_access_2[j]][0]
                    Price_cost_mc_2[m_c][t] += theta_generation_1[channel_access_2[j]][1]
                
            mixer = np.random.random(ucb_s_SU_3.size)
            ucb_indices_3 = list(np.lexsort((mixer, ucb_s_SU_3))) 
            # '''
            if ALGO == 'random':
                shuffle(ucb_indices_3)
            if ALGO == 'fixed':
                ucb_indices_3 = [8,7,6,5,4,3]
            # '''
            output_3 = ucb_indices_3[::-1]               
            channel_access_3 = output_3[ 2 * ((3+t) % U) : 2 * ((3+t) % U) + 2 ]
            for j in range(len(channel_access_3)):
                hat_theta_SU_3[channel_access_3[j]] = policy_SU_3[channel_access_3[j]].update_information(feature_generation, reward_SU_3[channel_access_3[j]])
                policy_SU_3[channel_access_3[j]].count_the_sensed_time()           
                Reward_all_simulation_mc_SU_3[m_c][t] += np.dot(feature_generation.T, theta_generation_1[channel_access_3[j]])
                if channel_state_3[channel_access_3[j]] == 1:
                    Throughput_mc_3[m_c][t] += channel_state_3[channel_access_3[j]] # theta_generation_1[channel_access_3[j]][0]
                    Price_cost_mc_3[m_c][t] += theta_generation_1[channel_access_3[j]][1] 
            
                    
            if t % 1000 == 0:
                print("this is the", t, "-th simulation")
                print(channel_access_1, channel_access_2, channel_access_3)
            
            collision_set_1_2 = list(set(channel_access_1) & set(channel_access_2))
            collision_set_2_3 = list(set(channel_access_2) & set(channel_access_3))
            collision_set_1_3 = list(set(channel_access_1) & set(channel_access_3))
            
            if collision_set_1_2 != []:
                # print("the collision_1_2 is", collision_set_1_2)
                for i in range(len(collision_set_1_2)):
                    Reward_all_simulation_mc_SU_1[m_c][t] = Reward_all_simulation_mc_SU_1[m_c][t] - np.dot(feature_generation.T, theta_generation_1[ collision_set_1_2[i] ])
                    Reward_all_simulation_mc_SU_2[m_c][t] = Reward_all_simulation_mc_SU_2[m_c][t] - np.dot(feature_generation.T, theta_generation_1[ collision_set_1_2[i] ])
                    collision_Num[m_c][t] += 1
                    Throughput_mc_1[m_c][t] = Throughput_mc_1[m_c][t] - channel_state[collision_set_1_2[i]] # theta_generation_1[collision_set_1_2[i]][0]
                    # Price_cost_mc_1[m_c][t] = Price_cost_mc_1[m_c][t] - channel_state[collision_set_1_2[i]] # theta_generation_1[collision_set_1_2[i]][1]
                    
            elif collision_set_2_3 != []:
                # print("the collision_2_3 is", collision_set_2_3)
                for i in range(len(collision_set_2_3)):
                    Reward_all_simulation_mc_SU_2[m_c][t] = Reward_all_simulation_mc_SU_2[m_c][t] - np.dot(feature_generation.T, theta_generation_1[ collision_set_2_3[i] ])
                    Reward_all_simulation_mc_SU_3[m_c][t] = Reward_all_simulation_mc_SU_3[m_c][t] - np.dot(feature_generation.T, theta_generation_1[ collision_set_2_3[i] ])
                    collision_Num[m_c][t] += 1
                    Throughput_mc_2[m_c][t] = Throughput_mc_2[m_c][t] - channel_state_2[collision_set_2_3[i]] # theta_generation_1[collision_set_2_3[i]][0]
                    # Price_cost_mc_2[m_c][t] = Price_cost_mc_2[m_c][t] - channel_state[collision_set_2_3[i]] # theta_generation_1[collision_set_2_3[i]][1]
                    
            elif collision_set_1_3 != []:
                # print("the collision_1_3 is", collision_set_1_3)
                for i in range(len(collision_set_1_3)):
                    Reward_all_simulation_mc_SU_1[m_c][t] = Reward_all_simulation_mc_SU_1[m_c][t] - np.dot(feature_generation.T, theta_generation_1[ collision_set_1_3[i] ])
                    Reward_all_simulation_mc_SU_3[m_c][t] = Reward_all_simulation_mc_SU_3[m_c][t] - np.dot(feature_generation.T, theta_generation_1[ collision_set_1_3[i] ])
                    collision_Num[m_c][t] += 1
                    Throughput_mc_3[m_c][t] = Throughput_mc_3[m_c][t] - channel_state_3[collision_set_1_3[i]] # theta_generation_1[collision_set_1_3[i]][0]
                    # Price_cost_mc_3[m_c][t] = Price_cost_mc_3[m_c][t] - channel_state[collision_set_1_3[i]] # theta_generation_1[collision_set_1_3[i]][1]           

                
            for element in range(Channel_Num):    
                ucb_s_SU_1[element] = policy_SU_1[element].compute_index(hat_theta_SU_1[element], feature_generation, policy_SU_1[element].invcov, t+1)
                ucb_s_SU_2[element] = policy_SU_2[element].compute_index(hat_theta_SU_2[element], feature_generation, policy_SU_2[element].invcov, t+1)
                ucb_s_SU_3[element] = policy_SU_3[element].compute_index(hat_theta_SU_3[element], feature_generation, policy_SU_3[element].invcov, t+1)
         
# In[8] plot the figure for regret and time slots 

t_x = list(range(T_simulation))
Reward_all_simulation_mc = Reward_all_simulation_mc_SU_1 + Reward_all_simulation_mc_SU_2 + Reward_all_simulation_mc_SU_3
reward_difference_mc = Reward_optimal_mc - Reward_all_simulation_mc

regret_x_label = [ round(0.005 * T_simulation), round(0.01 * T_simulation), round(0.03 * T_simulation), round(0.05 * T_simulation),  round(0.07 * T_simulation), round(0.1 * T_simulation), round(0.2 * T_simulation), round(0.3 *T_simulation), round(0.4 *T_simulation), round(0.5 *T_simulation), round(0.6 *T_simulation), round(0.7 *T_simulation), round(0.8 *T_simulation), round(0.9 *T_simulation),  T_simulation  ]


cumulative_regret_y_label   = np.zeros(len(regret_x_label))
accumulative_reward_y_label = np.zeros(len(regret_x_label))
cumulative_collision_y_label = np.zeros(len(regret_x_label))

for i,element in enumerate(regret_x_label):
    for j in range(reward_difference_mc.shape[0]):
        cumulative_regret_y_label[i] += sum(reward_difference_mc[j,0:element])
        accumulative_reward_y_label[i] += sum(Reward_all_simulation_mc[j][0:element])
        cumulative_collision_y_label[i] += sum(collision_Num[j][0:element])/m_c_number
    cumulative_regret_y_label[i] = cumulative_regret_y_label[i]/(m_c_number)
    accumulative_reward_y_label[i] = accumulative_reward_y_label[i]/(m_c_number*element)
    
plt.plot(regret_x_label, cumulative_regret_y_label,'b-^')
# plt.plot(regret_x_label, accumulative_reward_y_label)
plt.show()
 
# In[2] price and time slots 

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
plt.show()


# In[3] throughput and time slots 


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
# plt.ylim([0,2])
plt.show()




# In[4] bar: collisions and time slots 
fig, ax = plt.subplots()

t_x_bar = list(range(len(regret_x_label)))
# plt.bar(t_x_bar, cumulative_collision_y_label, color='tab:red')
plt.bar(t_x_bar, cumulative_collision_y_label)
ax = plt.gca()  #重新定义坐标系
tt = ['80', '160', '480', '800', '1120', '1600', '3200', '4800', '6400', '8000', '9600', '11200', '12800', '14400', '16000']
x = np.arange(len(regret_x_label))   #x轴的数据个数

ax.set_xticks(x)  #x轴坐标长度
ax.set_xticklabels(tt, rotation = 45)  
# ax.set_yticks([0, 50, 100, 150, 200, 250, 300])  
ax.set_xlabel("time slot")   #给X轴命名
ax.set_ylabel("The collision number")   
# fig.savefig('s_1_M_2.eps', dpi = 600, format = 'eps')
fig.savefig('s_2_M_2.eps', dpi = 600, format = 'eps')


# we record the 'cumulative_regret_y_label' (regret), 'average_throughput_1', 'average_throughput_2', 'average_throughput_3' for each user,
# record 'average_Price_1', 'average_Price_2', 'average_Price_3' for each user in 'plot_algorithm_price_QoS_Decentral.py'



