# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 08:41:00 2021 and modified on Aug 06 2023
for IEEE Transactions on Industrial Informatics

@author: xxx
feature_generation -- three QoS requirements

adjustable parameters:
set M = 2,3 to obtain simulation scenarios;
set TRAFFIC_TYPE = '1', '2', '3' to obtain various traffic types. 

we give the algorithm simulation for s=1. By setting TRAFFIC_TYPE = '1', '2', '3', we can obtain the corresponding performance.

We list the random, greedy, fixed algorithms as follows with annotations 

We put algorithms together including 'random', 'proposed', 'greedy', 'Fixed'.

We can set ALGO to obtain various simulation performance 

The results are recorded in 'plot_algorithm_single_user_price_QoS_various_probabilities.py'

"""

 
import numpy as np
from numpy.linalg import pinv
import math
import matplotlib.pyplot as plt
from my_SW_LinUCB_static import my_SW_LinUCB_static, my_SW_LinUCB_comparison
import random
from random import shuffle
import copy
# from softmax_standard import Softmax

# In[0] simulation parameters 
dimension = 2 # vector length
feature = np.zeros(dimension).reshape(dimension,1)
lambdada = 1
alpha = 1 # the algorithm parameter
transmission_rate = 1 # Mbps
transmission_duration = 1e-03 # s
m_c_number = 30 # Monte Carlo
M = 3   # the sensed channel number
K = M   # the accessed channel number
T_simulation = 10000 # the overall simulation time
T_init = 1           # the initial time
ALGO = 'proposed' # 'random', 'proposed', 'greedy', 'Fixed' ; various algorithms

# three w_s^T QoS requirements 
# s = 1 
feature_generation_1 = np.array([   [1], [0]  ]) 

# s = 2 
feature_generation_2 = np.array([   [0.9701], [0.2425]  ]) 

# s = 3 
feature_generation_3 = np.array([   [0.7071], [0.7071]  ]) 


# In[1] define the channel set
# [ idle probability, spectrum price ]
arm_theta_set_SU = [
np.array([0.85, -0.71]),
np.array([0.80, -0.83]),
np.array([0.75, -0.39]),
np.array([0.53, -0.45]),
np.array([0.23, -0.11]),
np.array([0.43, -0.35]),
np.array([0.48, -0.41]),
np.array([0.69, -0.21]), # 65
np.array([0.53, -0.11])  # 56
]

len_Num = len(arm_theta_set_SU)

arm_theta_set = arm_theta_set_SU
arm_theta_set = [ arm_theta_set[i].reshape(dimension,1) for i in range(len_Num) ]

Channel_Num = len(arm_theta_set)


# In[2] generate the policy set

policy_1 = []
policy_2 = []
policy_3 = []

policy = []
for k, arm_theta in enumerate(arm_theta_set):
    # policy.append(my_SW_LinUCB_static(arm_theta, feature, dimension, lambdada))
    policy_1.append(my_SW_LinUCB_comparison([], feature, dimension, lambdada))
    policy_2.append(my_SW_LinUCB_comparison([], feature, dimension, lambdada))
    policy_3.append(my_SW_LinUCB_comparison([], feature, dimension, lambdada))    
    
    
Reward_all_simulation = np.zeros(T_simulation)
Reward_optimal = np.zeros(T_simulation)
 
# In[6] Simulate more channels at one slot, there are N channels, M channels each slot

Reward = []
Reward_optimal_mc = np.zeros((m_c_number, T_simulation))
Reward_all_simulation_mc = np.zeros((m_c_number, T_simulation))
Throughput_mc = np.zeros((m_c_number, T_simulation))
Price_cost_mc = np.zeros((m_c_number, T_simulation))
reward_difference_mc = np.zeros((m_c_number, T_simulation))

Reward_1_optimal_mc = np.zeros((m_c_number, T_simulation))
Reward_2_optimal_mc = np.zeros((m_c_number, T_simulation))
Reward_3_optimal_mc = np.zeros((m_c_number, T_simulation))

Reward_all_simulation_mc_1 = np.zeros((m_c_number, T_simulation))
Throughput_mc_1 = np.zeros((m_c_number, T_simulation))
Price_cost_mc_1 = np.zeros((m_c_number, T_simulation))


Reward_all_simulation_mc_2 = np.zeros((m_c_number, T_simulation))
Throughput_mc_2 = np.zeros((m_c_number, T_simulation))
Price_cost_mc_2 = np.zeros((m_c_number, T_simulation)) 


Reward_all_simulation_mc_3 = np.zeros((m_c_number, T_simulation))
Throughput_mc_3 = np.zeros((m_c_number, T_simulation))
Price_cost_mc_3 = np.zeros((m_c_number, T_simulation))
 
for m_c in range(m_c_number):
       
    print("This is the", m_c, "simulation")
    for i in range(len(policy_1)):
        policy_1[i].re_init() # initialization
        policy_2[i].re_init() # initialization        
        policy_3[i].re_init() # initialization        
   
    hat_theta_one = np.zeros((dimension,1)).reshape(dimension,1)  # how to fastly generate the list

    hat_theta_1 = []
    hat_theta_2 = []
    hat_theta_3 = []
    
    for i in range(Channel_Num):
        hat_theta_1.append(hat_theta_one)
        hat_theta_2.append(hat_theta_one)
        hat_theta_3.append(hat_theta_one)   
        

    ucb_s_1 = np.zeros(Channel_Num)
    ucb_s_2 = np.zeros(Channel_Num)
    ucb_s_3 = np.zeros(Channel_Num)
        
 
    
    theta_generation = copy.deepcopy(arm_theta_set)     
    estimated_reward_greedy = np.zeros(Channel_Num)
    number_of_trials = np.zeros(Channel_Num)  
    channel_state = np.zeros(Channel_Num) # record the channel states
    
    # ----------------- epsilon-greedy ------------------ #
    # epsilon_prob = 0.05 
    # estimated_reward_greedy = np.zeros(len(arm_theta_set))
    # number_of_trials = np.zeros(len(arm_theta_set))    
    # # --------------------------------------------------- #
    # Softmax_algo = Softmax(1, Channel_Num)
    
    # --------------------------------------------------- #
    # Q_ = np.random.rand(Channel_Num)
    # Q_ = np.zeros(Channel_Num)
    
    # count = np.zeros(Channel_Num)
 
    for t in range(T_simulation):
 

        # instantaneous channel state
        Reward_1 = []
        Reward_2 = []
        Reward_3 = []
        for i, theta_index in enumerate(arm_theta_set):
            channel_state[i] = copy.deepcopy(np.random.binomial(1,theta_generation[i][0],1)) # instantanesou channel states
            Reward_1.append( feature_generation_1[0] * channel_state[i] + feature_generation_1[1] * theta_generation[i][1] )
            Reward_2.append( feature_generation_2[0] * channel_state[i] + feature_generation_2[1] * theta_generation[i][1] )            
            Reward_3.append( feature_generation_3[0] * channel_state[i] + feature_generation_3[1] * theta_generation[i][1] )            
                
        if t < T_init: # initialization, each channel has been sensed
            for i, theta_index in enumerate(arm_theta_set):    
                hat_theta_1[i] = policy_1[i].update_information(feature_generation_1, Reward_1[i])
                ucb_s_1[i] = policy_1[i].compute_index(hat_theta_1[i], feature_generation_1, policy_1[i].invcov, t)

                hat_theta_2[i] = policy_2[i].update_information(feature_generation_2, Reward_2[i])
                ucb_s_2[i] = policy_2[i].compute_index(hat_theta_2[i], feature_generation_2, policy_2[i].invcov, t)
                
                hat_theta_3[i] = policy_3[i].update_information(feature_generation_3, Reward_3[i])
                ucb_s_3[i] = policy_3[i].compute_index(hat_theta_3[i], feature_generation_3, policy_3[i].invcov, t)
                
                
        else:

            # order the channel ucb_s to output M channels, which used for accessing selection
    
   
            thre_1 = 0.3
            thre_2 = 0.6
            T_thres_1 = 2000
            T_thres_2 = 6000
            random_prob = np.random.random()
            
    # In[1]            
            # two scenarios : 
                # scenario I :
            if random_prob < thre_1:
                '''
                # if we want to change to the scenario II, we can delete the annotation as follows
                # scenario II :
            # if t < T_thres_1:
                '''
                # update the channel information
                for element in range(Channel_Num):                
                    ucb_s_1[element] = policy_1[element].compute_index(hat_theta_1[element], feature_generation_1, policy_1[element].invcov, t)

                mixer_1 = np.random.random(ucb_s_1.size)
                ucb_indices_1 = list(np.lexsort((mixer_1, ucb_s_1))) 
                output_1 = ucb_indices_1[::-1]
                
                # In[1] for various algorithms
                if ALGO == 'random':
                    # print('this is the random algo')
                    shuffle(output_1)
                elif ALGO == 'greedy':
                    if np.random.random() < 0.1:
                        shuffle(output_1)
                elif ALGO == 'Fixed':
                    output_1 = copy.deepcopy([4,7,8])  
                else:
                    pass    
                chosen_arm_1 = output_1[0:M]  # M channels are chosen to be sensed   
 
 
                for i, element in enumerate(chosen_arm_1):    
                    hat_theta_1[element] = policy_1[element].update_information(feature_generation_1, Reward_1[element])
                    policy_1[element].count_the_sensed_time()
                    
                # idle channel set
                arm_access_chosen_1 = []
                for element in chosen_arm_1:
                    if channel_state[element] == 1: # if channel sensed idle
                        arm_access_chosen_1.append(element)
                    
                for element in chosen_arm_1:  
                    Reward_all_simulation_mc_1[m_c][t] += copy.deepcopy( Reward_1[ element ] )  # np.dot(feature_generation.T, theta_generation[arm_access_chosen[i]]) 
                    if channel_state[element] == 1: # if channel sensed idle, then access
                        Throughput_mc_1[m_c][t] += 1 # * transmission_rate * transmission_duration  
                        Price_cost_mc_1[m_c][t] += theta_generation[element][1]                       
                    
                # the benchmark    
                Reward = []
                for i, theta_index in enumerate(arm_theta_set):
                    Reward.append( np.dot(feature_generation_1.T, theta_generation[i])  ) 
                Reward_1_optimal_mc[m_c, t] = sum(sorted(Reward, reverse = True)[0:M])                    
 
    # In[2]
            # two scenarios : 
                # scenario I : 
            elif thre_1 <= random_prob < thre_2:
                # scenario II :
            # elif T_thres_1 < t < T_thres_2:  
                # update the channel information
                for element in range(Channel_Num):                
                    ucb_s_2[element] = policy_2[element].compute_index(hat_theta_2[element], feature_generation_2, policy_2[element].invcov, t)

                mixer_2 = np.random.random(ucb_s_2.size)
                ucb_indices_2 = list(np.lexsort((mixer_2, ucb_s_2))) 
                output_2 = ucb_indices_2[::-1]
                if ALGO == 'random':
                    # print('this is the random algo')
                    shuffle(output_2)
                elif ALGO == 'greedy':
                    if np.random.random() < 0.1:
                        shuffle(output_2) 
                elif ALGO == 'Fixed':
                    output_2 = copy.deepcopy([4,7,8])
                else:
                    pass
                chosen_arm_2 = output_2[0:M]  # M channels are chosen to be sensed   
 
                for i, element in enumerate(chosen_arm_2):    
                    hat_theta_2[element] = policy_2[element].update_information(feature_generation_2, Reward_2[element])
                    policy_2[element].count_the_sensed_time()
                    
                # idle channel set
                arm_access_chosen_2 = []
                for element in chosen_arm_2:
                    if channel_state[element] == 1: # if channel sensed idle
                        arm_access_chosen_2.append(element)
                    
                for element in chosen_arm_2:  
                    Reward_all_simulation_mc_2[m_c][t] += copy.deepcopy( Reward_2[ element ] )  # np.dot(feature_generation.T, theta_generation[arm_access_chosen[i]]) 
                    if channel_state[element] == 1: # if channel sensed idle, then access
                        Throughput_mc_2[m_c][t] += 1 # * transmission_rate * transmission_duration  
                        Price_cost_mc_2[m_c][t] += theta_generation[element][1]                       
                    
                # the benchmark    
                Reward = []
                for i, theta_index in enumerate(arm_theta_set):
                    Reward.append( np.dot(feature_generation_2.T, theta_generation[i])  ) 
                Reward_2_optimal_mc[m_c, t] = sum(sorted(Reward, reverse = True)[0:M]) 
                
     # In[3]        
            else:        
                for element in range(Channel_Num):                
                    ucb_s_3[element] = policy_3[element].compute_index(hat_theta_3[element], feature_generation_3, policy_3[element].invcov, t)

                mixer_3 = np.random.random(ucb_s_3.size)
                ucb_indices_3 = list(np.lexsort((mixer_3, ucb_s_3))) 
                output_3 = ucb_indices_3[::-1]
                if ALGO == 'random':
                    # print('this is the random algo')
                    shuffle(output_3)
                elif ALGO == 'greedy':
                    if np.random.random() < 0.1:
                        shuffle(output_3)
                elif ALGO == 'Fixed':
                    output_3 = copy.deepcopy([4,7,8])
                else:
                    pass
                chosen_arm_3 = output_3[0:M]  # M channels are chosen to be sensed   
 
                for i, element in enumerate(chosen_arm_3):    
                    hat_theta_3[element] = policy_3[element].update_information(feature_generation_3, Reward_3[element])
                    policy_3[element].count_the_sensed_time()
                    
                # idle channel set
                arm_access_chosen_3 = []
                for element in chosen_arm_3:
                    if channel_state[element] == 1: # if channel sensed idle
                        arm_access_chosen_3.append(element)
                    
                for element in chosen_arm_3:  
                    Reward_all_simulation_mc_3[m_c][t] += copy.deepcopy( Reward_3[ element ] )  # np.dot(feature_generation.T, theta_generation[arm_access_chosen[i]]) 
                    if channel_state[element] == 1: # if channel sensed idle, then access
                        Throughput_mc_3[m_c][t] += 1 # * transmission_rate * transmission_duration  
                        Price_cost_mc_3[m_c][t] += theta_generation[element][1]                       
                    
                # the benchmark    
                Reward = []
                for i, theta_index in enumerate(arm_theta_set):
                    Reward.append( np.dot(feature_generation_3.T, theta_generation[i])  ) 
                Reward_3_optimal_mc[m_c, t] = sum(sorted(Reward, reverse = True)[0:M]) 

        
# In[8] plot the figure 

t_x = list(range(T_simulation))
reward_difference_mc = Reward_1_optimal_mc - Reward_all_simulation_mc_1 + Reward_2_optimal_mc - Reward_all_simulation_mc_2 + Reward_3_optimal_mc - Reward_all_simulation_mc_3 


# regret_x_label = [10,  50, 55, 75, 80,85, 90 ,100,150, 200,250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 10500, 11000, 12000, 18000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 100000 ]
# regret_x_label = [ 100, 300, 500, 1000,  1500,  2000, 4000, 6000, 8000, 10000, 15000, 20000  ]
regret_x_label = [ round(0.03 * T_simulation), round(0.05 * T_simulation), round(0.1 * T_simulation), round(0.2 * T_simulation), round(0.3 *T_simulation), round(0.4 *T_simulation), round(0.5 *T_simulation), round(0.6 *T_simulation), round(0.7 *T_simulation), round(0.8 *T_simulation), round(0.9 *T_simulation),  T_simulation  ]
# regret_x_label = [ round(0.03 * T_simulation), round(0.05 * T_simulation), round(0.1 * T_simulation),round(0.15 * T_simulation) ]

cumulative_regret_y_label   = np.zeros(len(regret_x_label))
accumulative_reward_y_label = np.zeros(len(regret_x_label))


for i,element in enumerate(regret_x_label):
    for j in range(reward_difference_mc.shape[0]):
        cumulative_regret_y_label[i] += sum(reward_difference_mc[j,0:element])
    cumulative_regret_y_label[i] = cumulative_regret_y_label[i]/(m_c_number  )
 
    
plt.plot(regret_x_label, cumulative_regret_y_label,'--*')
plt.title('regret and time slots')
# plt.plot(regret_x_label, accumulative_reward_y_label)
plt.show()
 

chosen_time_matrix_1 = []
chosen_time_matrix_2 = []
chosen_time_matrix_3 = []

for n in range(len_Num):
    chosen_time_matrix_1.append( policy_1[n].t_count )
    chosen_time_matrix_2.append( policy_2[n].t_count )
    chosen_time_matrix_3.append( policy_3[n].t_count )



Throuhgput = sum( Throughput_mc_1 + Throughput_mc_2 + Throughput_mc_3 )/m_c_number
Price = sum( Price_cost_mc_1 + Price_cost_mc_2 + Price_cost_mc_3 )/m_c_number

Throughput_all = sum(Throuhgput)
Price_all = sum(Price)

# record the throughput
cumulative_throuhgput_y_label   = np.zeros(len(regret_x_label))
average_throughput = np.zeros(len(regret_x_label))

# record the cost
cumulative_cost_y_label   = np.zeros(len(regret_x_label))
average_cost = np.zeros(len(regret_x_label))

for i,element in enumerate(regret_x_label):
    cumulative_throuhgput_y_label[i] += sum(Throuhgput[0:element])
    average_throughput[i] = cumulative_throuhgput_y_label[i]/element
 
    cumulative_cost_y_label[i] += sum(Price[0:element])
    average_cost[i] = -1 * cumulative_cost_y_label[i]/element


# plt.plot(regret_x_label, cumulative_throuhgput_y_label,'b-^')
# plt.plot(regret_x_label, average_throughput,'b-^')

 
# plt.plot(regret_x_label, average_cost,'b-^')

print('The regret is', cumulative_regret_y_label)
print('The average_throughput is', average_throughput)
print('The average_cost is', average_cost)


# we finally record 'cumulative_regret_y_label', 'average_throughput', and 'average_cost' in 'plot_algorithm_single_user_price_QoS_various_probabilities.py'






