# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 08:41:00 2021, modified on Aug 8 2023.
for IEEE Transactions on Industrial Informatics

@author: xxx

TRAFFIC_TYPE = '1' # '2' various QoS requirements

we record the results in plot_algorithm_single_user_energy_throughput.py' 

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
from random import shuffle
# from softmax_standard import Softmax

# In[0] simulation parameters 
dimension = 3
feature = np.zeros(dimension).reshape(dimension,1)
lambdada = 2
alpha = 1 # the algorithm parameter
m_c_number = 5 # Monte Carlo Number 
duration = 95e-03 # ms 
transmission_rate = 1 # Mbps
Bandwidth_set = [0.6, 0.6, 0.6, 0.8, 0.8, 0.8, 1, 1, 1] # bandwidth
T_simulation = 5000   # the overall simulation time
T_init = 1            # the initial verification time
ALGO = 'fixed' #  'fixed' 'random' 'proposed'. select various algorithms 
M = 3   # the sensed channel number
K = M   # the accessed channel number


TRAFFIC_TYPE = '2' # '2' 

if TRAFFIC_TYPE == '1':
    feature_generation = np.array([   [0.707], [0.707], [0.14]  ])
elif TRAFFIC_TYPE == '2':
    feature_generation = np.array([   [0.3015], [0.3015], [0.9045]  ])

# 9 条信道 [空闲概率, SER, Transmit_Power]，在后续部分添加
arm_theta_set_SU = [
np.array([0.85, 0, 0]),
np.array([0.80, 0, 0]),
np.array([0.75, 0, 0]),
np.array([0.53, 0, 0]),
np.array([0.23, 0, 0]),
np.array([0.43, 0, 0]),
np.array([0.48, 0, 0]),
np.array([0.59, 0, 0]),
np.array([0.53, 0, 0])
]

len_Num = len(arm_theta_set_SU)

# In[0] new parameter settings
transmission_distance = 200
Tx_antenna_Num = 4

def PathLoss_(transmission_distance, Tx_antenna_Num, Channel_Noise):
    # channel noise is in the form of num
    PathLoss_Model_Parameter = np.sqrt( 10**(-3.53)/( transmission_distance**(3.76)) )
    Channel_parameter_vector = []
    for i in range( Tx_antenna_Num ):
        Channel_parameter_vector.append( PathLoss_Model_Parameter * 1/np.sqrt(2) * ( np.random.normal() + complex(0, 1) * np.random.normal() ) )
        # 其他场景直接求信噪比比较好 或者信道系数 Rician
        # Channel_parameter_vector.append( PathLoss_Model_Parameter * ( np.sqrt( K_factor/(K_factor+1) ) * ( np.random.normal() + complex(0, 1) * np.random.normal() ) + np.sqrt( 1/(K_factor+1) ) * np.random.normal()   )  )
        
    Channel_parameter_norm = np.linalg.norm( Channel_parameter_vector )
    Channel_gain_noise_ratio = Channel_parameter_norm**2/Channel_Noise
    return Channel_gain_noise_ratio

# channel noise
Channel_Noise_vector = [ -94 - 1 * i for i in range(len_Num) ]
Channel_Noise = [ 10**(Channel_Noise_vector[i]/10)/(10**3) for i in range(len(Channel_Noise_vector)) ] 
 


def Power_minimum_(transmission_rate, Bandwidth, Channel_gain_noise_ratio, duration):
    Power_minimum = ( 2**(transmission_rate/Bandwidth) - 1 )/Channel_gain_noise_ratio
    return Power_minimum

def Energy_minimum_(Power_minimum, duration): # definition of the network cost
    # Power_minimum = ( 2**(transmission_rate/Bandwidth) - 1 )/Channel_gain_noise_ratio
    Energy_minimum = Power_minimum * duration
    return Energy_minimum    

def symbol_error_rate_(power_transmit, Channel_gain_noise_ratio):
    SER = np.e**(-power_transmit * Channel_gain_noise_ratio)
    return SER
 
# In[1] define the channel set
 
arm_theta_set = arm_theta_set_SU
arm_theta_set = [arm_theta_set[i].reshape(dimension,1) for i in range(len_Num)]

Channel_Num = len(arm_theta_set)


# In[2] generate the policy set
policy = []
for k, arm_theta in enumerate(arm_theta_set):
    # policy.append(my_SW_LinUCB_static(arm_theta, feature, dimension, lambdada))
    policy.append(my_SW_LinUCB_comparison([], feature, dimension, lambdada))

Reward_all_simulation = np.zeros(T_simulation)
Reward_optimal = np.zeros(T_simulation)
 
# In[6] Simulate more channels at one slot, there are N channels, M channels each slot

Reward = []
Reward_optimal_mc = np.zeros((m_c_number, T_simulation))
Reward_all_simulation_mc = np.zeros((m_c_number, T_simulation))
Throughput_mc = np.zeros((m_c_number, T_simulation))
Energy_cost_mc = np.zeros((m_c_number, T_simulation))
reward_difference_mc = np.zeros((m_c_number, T_simulation))

for m_c in range(m_c_number):
    
    print("This is the", m_c, "simulation")
    for i, policy_index in enumerate(policy):
        policy[i].re_init()
        
    reward = np.zeros(Channel_Num)
    hat_theta_one = np.zeros((dimension,1)).reshape(dimension,1)  # how to fastly generate the list
    hat_theta = []
    for i in range(Channel_Num):
        hat_theta.append(hat_theta_one)
    ucb_s = np.zeros(Channel_Num)
    Channel_gain_noise_ratio_set = np.zeros(Channel_Num)
    Power_transmit_set = np.zeros(Channel_Num)
    Energy_transmit_set = np.zeros(Channel_Num)
    SER_set = np.zeros(Channel_Num)
    
    theta_generation = copy.deepcopy(arm_theta_set) 
    
    estimated_reward_greedy = np.zeros(Channel_Num)
    number_of_trials = np.zeros(Channel_Num)  
 
    count = np.zeros(Channel_Num)
    channel_state = np.zeros(Channel_Num) # 信道状态
   
    for t in range(T_simulation):
 
        # 产生信道特征 # 每条信道的 H^2/sigma^2
        Channel_gain_noise_ratio_set = [ PathLoss_(transmission_distance, Tx_antenna_Num, Channel_Noise[i]) for i in range(len_Num) ]
        Power_transmit_set = [ Power_minimum_(transmission_rate, Bandwidth_set[i], Channel_gain_noise_ratio_set[i], duration) for i in range(len_Num) ] 
        Energy_transmit_set = [ -1 * Energy_minimum_(Power_transmit_set[i], duration) for i in range(len_Num) ]
        SER_set = [ 1 - symbol_error_rate_(Power_transmit_set[i], Channel_gain_noise_ratio_set[i]) for i in range(len_Num) ]
 


        for i, theta_index in enumerate(arm_theta_set):  
            channel_state[i] = np.random.binomial(1,theta_generation[i][0],1)
            reward[i] = feature_generation[0] * channel_state[i] + feature_generation[1] * SER_set[i] + feature_generation[2] * Energy_transmit_set[i]    
        
 
        if t < T_init: # initialization, each channel has been sensed
            for i, theta_index in enumerate(arm_theta_set):
                # reward[i] = feature_generation[0]*np.random.binomial(1,theta_generation[i][0],1) + np.dot(feature_generation[-(len(feature_generation)-1):-1].T, theta_generation[i][-(len(feature_generation)-1):-1]) + feature_generation[-1]*np.random.binomial(1,theta_generation[i][-1],1)              
                # reward[i] = feature_generation[0]*np.random.binomial(1,theta_generation[i][0],1) + np.dot(feature_generation[-(len(feature_generation)-1):].T, theta_generation[i][-(len(feature_generation)-1):])
                # reward[i] = feature_generation[0]*np.random.binomial(1,theta_generation[i][0],1) + feature_generation[1]*theta_generation[i][1] + feature_generation[2] * np.random.normal(theta_generation[i][2],0.1) + feature_generation[3] * theta_generation[i][3] + feature_generation[4] * theta_generation[i][4]        
                
                hat_theta[i] = policy[i].update_information(feature_generation, reward[i])
                ucb_s[i] = policy[i].compute_index(hat_theta[i], feature_generation, policy[i].invcov, t)
        else:
            for element in range(Channel_Num):                
                ucb_s[element] = policy[element].compute_index(hat_theta[element], feature_generation, policy[element].invcov, t)
            
            # order the channel ucb_s to output M channels 
            mixer = np.random.random(ucb_s.size)
            ucb_indices = list(np.lexsort((mixer, ucb_s))) 
            output = ucb_indices[::-1]
            chosen_arm = output[0:M]  # M channels are chosen to be sensed 
            
            if ALGO == 'random':
                shuffle(output)
                chosen_arm = output[0:M]
            elif ALGO == 'fixed':
                chosen_arm = [6,7,8]
 
            # update the channel information
            for i, element in enumerate(chosen_arm):    
                hat_theta[element] = policy[element].update_information(feature_generation, reward[element])
                policy[element].count_the_sensed_time()
                
            for element in chosen_arm:
                Reward_all_simulation_mc[m_c][t] += copy.deepcopy( reward[ element ] )  # np.dot(feature_generation.T, theta_generation[arm_access_chosen[i]]) 
                if channel_state[element] == 1: # 如果信道被感知为空闲，则接入
                    Throughput_mc[m_c][t] += 1 * SER_set[element] * duration # 因为空闲，所以传输成功 1 Mbps * 成功率 * 传输时长 95e-03 s 
                    Energy_cost_mc[m_c][t] += Energy_transmit_set[element]  

        # the benchmark 
        # if feature_generation = np.array([   [0.707], [0.707], [0.14]  ]), then the optimal channel set is [0,1,2]
        # if feature_generation = np.array([   [0.3015], [0.3015], [0.9045]]), then the optimal channel set is [6,7,8]
        Reward = []
        for i, theta_index in enumerate(arm_theta_set):
            Reward.append( np.dot(feature_generation.T, theta_generation[i])  ) 
        Reward_optimal_mc[m_c, t] = sum(sorted(Reward, reverse = True)[0:M])
        
        
# In[8] plot the figure for throughput and energy consumption  

t_x = list(range(T_simulation))
reward_difference_mc = Reward_optimal_mc - Reward_all_simulation_mc

# regret_x_label = [10,  50, 55, 75, 80,85, 90 ,100,150, 200,250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 10500, 11000, 12000, 18000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 100000 ]
# regret_x_label = [ 100, 300, 500, 1000,  1500,  2000, 4000, 6000, 8000, 10000, 15000, 20000  ]
regret_x_label = [ round(0.1 * T_simulation), round(0.2 * T_simulation), round(0.3 *T_simulation), round(0.4 *T_simulation), round(0.5 *T_simulation), round(0.6 *T_simulation), round(0.7 *T_simulation), round(0.8 *T_simulation), round(0.9 *T_simulation),  T_simulation  ]


cumulative_regret_y_label   = np.zeros(len(regret_x_label))
accumulative_reward_y_label = np.zeros(len(regret_x_label))


for i,element in enumerate(regret_x_label):
    for j in range(reward_difference_mc.shape[0]):
        cumulative_regret_y_label[i] += sum(reward_difference_mc[j,0:element])
        accumulative_reward_y_label[i] += sum(Reward_all_simulation_mc[j][0:element])
    cumulative_regret_y_label[i] = cumulative_regret_y_label[i]/(m_c_number  )
    accumulative_reward_y_label[i] = accumulative_reward_y_label[i]/(m_c_number*element)
    
 
# for i in range(9):
# cumulative_regret_y_label    print(policy[i].t_count)

chosen_time_matrix = []
for n in range(len_Num):
    chosen_time_matrix.append( policy[n].t_count )

Throuhgput = sum(Throughput_mc)/m_c_number
Energy_cost = sum(Energy_cost_mc)/m_c_number

Throughput_all = sum(Throuhgput)
Energy_all = sum(Energy_cost)

# record the throughput
# cumulative_throuhgput_y_label   = np.zeros(len(regret_x_label))
# average_throughput = np.zeros(len(regret_x_label))


print('The cumulative throughput is', Throughput_all)
print('The cumulative energy consumption is', Energy_all)
print('The chosen time is', chosen_time_matrix)


Throuhgput_y_label   = np.zeros(len(regret_x_label))
Energy_cost_y_label = np.zeros(len(regret_x_label))

for i,element in enumerate(regret_x_label):
    for j in range(reward_difference_mc.shape[0]):
        Throuhgput_y_label[i] += sum(Throughput_mc[j,0:element])
        Energy_cost_y_label[i] += sum(-Energy_cost_mc[j,0:element])
        # accumulative_reward_y_label[i] += sum(Reward_all_simulation_mc[j][0:element])
    Throuhgput_y_label[i] = Throuhgput_y_label[i]/(m_c_number  )
    Energy_cost_y_label[i] = Energy_cost_y_label[i]/(m_c_number  )
    # accumulative_reward_y_label[i] = accumulative_reward_y_label[i]/(m_c_number*element)



# plt.bar(regret_x_label, Energy_cost_y_label)

 
# plt.bar(regret_x_label, Throuhgput_y_label)

cumulative_throuhgput_y_label = np.zeros(len(regret_x_label))
average_throughput = np.zeros(len(regret_x_label))
cumulative_energy_y_label = np.zeros(len(regret_x_label))
average_energy = np.zeros(len(regret_x_label))

for i,element in enumerate(regret_x_label):
    cumulative_throuhgput_y_label[i] += sum(Throuhgput[0:element])
    average_throughput[i] = cumulative_throuhgput_y_label[i]/element
    cumulative_energy_y_label[i] += sum(Energy_cost[0:element])
    average_energy[i] = cumulative_energy_y_label[i]/element

# plt.plot(regret_x_label, average_throughput,'b-^')


# plt.plot(regret_x_label, average_energy,'b-^')



print("不同时刻为", regret_x_label)
print("不同时刻下累积吞吐量为", Throuhgput_y_label)
print("不同时刻下累积能耗为", Energy_cost_y_label)


# we record Throuhgput_y_label and Energy_cost_y_label in 'plot_algorithm_single_user_energy_throughput.py' file 

 


















