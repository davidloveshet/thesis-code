# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 08:41:00 2021 and modified on Aug 06 2023
for IEEE Transactions on Industrial Informatics

@author: xxx
feature_generation -- three QoS requirement

adjustable parameters:
set M = 2,3 to obtain simulation scenarios;
set TRAFFIC_TYPE = '1', '2', '3' to obtain various traffic types. 

we give the algorithm simulation for s=1. By setting TRAFFIC_TYPE = '1', '2', '3', we can obtain the corresponding performance.

We list the random, greedy, fixed algorithms as follows with annotations 

The results are recorded in 'plot_algorithm_single_user_price_QoS.py'
 
"""
 
import numpy as np
from numpy.linalg import pinv
import math
import matplotlib.pyplot as plt
from my_SW_LinUCB_static import my_SW_LinUCB_static, my_SW_LinUCB_comparison
from random import shuffle
import copy
# from softmax_standard import Softmax

# In[0] simulation parameter
dimension = 2 # dimension of QoS requirement 
feature = np.zeros(dimension).reshape(dimension,1)
lambdada = 1
transmission_rate = 1 # Mbps
transmission_duration = 1e-03 # s
m_c_number = 30 # Monte_Carlo Number 
M = 3   # the sensed channel number
K = M   # the accessed channel number
T_simulation = 10000 # the overall simulation time
T_init = 1           # the initial time
 
# three w_s^T QoS requirement 
TRAFFIC_TYPE = '3' # '1', '2', '3'

if TRAFFIC_TYPE == '1': # the optimal channels are {1,2,3}
    feature_generation = np.array([   [1], [0]  ]) 
elif TRAFFIC_TYPE == '2': # the optimal channels are {1,3,8}
    feature_generation = np.array([   [0.9701], [0.2425]  ]) 
elif TRAFFIC_TYPE == '3': # the optimal channels are {3,8,9}
    feature_generation = np.array([   [0.7071], [0.7071]  ])     
   
    
ALGO = 'greedy' # 'random', 'fixed', 'greedy', 'proposed'. Select a setting, we can obtain the algorithm result 


   
    
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
np.array([0.69, -0.21]),  
np.array([0.53, -0.11])   
]

len_Num = len(arm_theta_set_SU) 

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
 
# In[6] Simulation
print("The channel sensing number is", M)
print("The Traffic type is", TRAFFIC_TYPE)

Reward = []
Reward_optimal_mc = np.zeros((m_c_number, T_simulation))
Reward_all_simulation_mc = np.zeros((m_c_number, T_simulation))
Throughput_mc = np.zeros((m_c_number, T_simulation))
Price_cost_mc = np.zeros((m_c_number, T_simulation))
reward_difference_mc = np.zeros((m_c_number, T_simulation))

for m_c in range(m_c_number):
    
    print("This is the", m_c, "simulation")
    for i, policy_index in enumerate(policy):
        policy[i].re_init() # initialization
    reward = np.zeros(Channel_Num)
    hat_theta_one = np.zeros((dimension,1)).reshape(dimension,1)  # how to fastly generate the list
    hat_theta = []
    for i in range(Channel_Num):
        hat_theta.append(hat_theta_one)
    ucb_s = np.zeros(Channel_Num)   # upper confidence bound of the estimated utility reward 
    
    theta_generation = copy.deepcopy(arm_theta_set)     
    estimated_reward_greedy = np.zeros(Channel_Num)
    number_of_trials = np.zeros(Channel_Num)  
    channel_state = np.zeros(Channel_Num) # channel state 
    
    # ----------------- epsilon-greedy ------------------ #
    # epsilon_prob = 0.05 
    # estimated_reward_greedy = np.zeros(len(arm_theta_set))
    # number_of_trials = np.zeros(len(arm_theta_set))    
    # # -------------- Softmax algorithm ---------------- #
    # Softmax_algo = Softmax(1, Channel_Num)
    # --------------------------------------------------- #
    # Q_ = np.random.rand(Channel_Num)
    # Q_ = np.zeros(Channel_Num)
    # count = np.zeros(Channel_Num)
 
    for t in range(T_simulation):
 
        # instantaneous channel state
        for i, theta_index in enumerate(arm_theta_set):
            channel_state[i] = copy.deepcopy(np.random.binomial(1,theta_generation[i][0],1)) # instantaneous channel state 
            reward[i] = feature_generation[0] * channel_state[i] + feature_generation[1] * theta_generation[i][1]              
                
        if t < T_init: # initialization, each channel has been sensed
            for i, theta_index in enumerate(arm_theta_set):    
                hat_theta[i] = policy[i].update_information(feature_generation, reward[i])
                ucb_s[i] = policy[i].compute_index(hat_theta[i], feature_generation, policy[i].invcov, t+1) # upper confidence bound of the estimated utility reward
        else: 
            
            # '''
            # proposed algorithm 
            for element in range(Channel_Num):                
                ucb_s[element] = policy[element].compute_index(hat_theta[element], feature_generation, policy[element].invcov, t)
                
            # order the channel ucb_s to output M channels, which used for accessing selection
            
            # In[1] the greedy arm  
            epsilon_prob = 0.1
            if ALGO == 'greedy':
                if np.random.random() < epsilon_prob:
                    ucb_s = np.random.random(Channel_Num)                
           
            mixer = np.random.random(ucb_s.size)
            ucb_indices = list(np.lexsort((mixer, ucb_s))) 
            output = ucb_indices[::-1]
            chosen_arm = output[0:M]  # M channels are chosen to be sensed  
            # '''
            
            # In[2] the fixed chosen arm 
            if ALGO == 'fixed':
                chosen_arm = [4,7,8]  # fixed channel chosen             
 
            
            # In[3] random algorithm
            if ALGO == 'random':
                shuffle(output)
                chosen_arm = copy.deepcopy(output[0:M])  # M channels are chosen to be sensed  
 
             
            for i, element in enumerate(chosen_arm):    
                hat_theta[element] = policy[element].update_information(feature_generation, reward[element])
                policy[element].count_the_sensed_time()
 
            # idle channel set 
            arm_access_chosen = []
            for element in chosen_arm:
                if channel_state[element] == 1: # sensed idle 
                    arm_access_chosen.append(element)
 
            for element in chosen_arm:
                Reward_all_simulation_mc[m_c][t] += copy.deepcopy( reward[ element ] )  # np.dot(feature_generation.T, theta_generation[arm_access_chosen[i]]) 
                if channel_state[element] == 1: # if channels are sensed idle, access
                    Throughput_mc[m_c][t] += 1 # * transmission_rate * transmission_duration # since we consider the average throughput, we record the success number 
                    Price_cost_mc[m_c][t] += theta_generation[element][1] # price        
        
        # ------------------------------------- other algorithms ------------------------------------- #
        '''
        # annealing epsilon
        if t%200 == 0:
            epsilon_prob = epsilon_prob/np.log(np.e+0.3)
        # epsilon-greedy algorithm
        if t < T_init:
            for i, theta_index in enumerate(arm_theta_set):
                estimated_reward_greedy[i] = feature_generation[0]*np.random.binomial(1,theta_generation[i][0],1) + feature_generation[1]*theta_generation[i][1] + feature_generation[2] * np.random.normal(theta_generation[i][2],0.1) + feature_generation[3] * theta_generation[i][3] + feature_generation[4] * theta_generation[i][4]       
            ucb_s = copy.deepcopy(estimated_reward_greedy)
        else:
            mixer = np.random.random(ucb_s.size)
            ucb_indices = list(np.lexsort((mixer, ucb_s))) 
            output = ucb_indices[::-1]
            chosen_arm = output[0:M]  # M channels are chosen to be sensed 
            for i in range(Channel_Num):
                reward[i] = feature_generation[0]*np.random.binomial(1,theta_generation[i][0],1) + feature_generation[1]*theta_generation[i][1] + feature_generation[2] * np.random.normal(theta_generation[i][2],0.1) + feature_generation[3] * theta_generation[i][3] + feature_generation[4] * theta_generation[i][4]     
            if np.random.random() < epsilon_prob:
                chosen_arm = np.random.randint(0, Channel_Num, M)      
            for i, element in enumerate(chosen_arm):
                number_of_trials[element] = number_of_trials[element] + 1
                estimated_reward_greedy[element] = (estimated_reward_greedy[element] * (number_of_trials[element] - 1) + reward[element])/number_of_trials[element]                
            mixer_ = np.random.random(estimated_reward_greedy.size)
            ucb_indices_ = list(np.lexsort((mixer, estimated_reward_greedy))) 
            output_ = ucb_indices[::-1]
            arm_access_chosen = output[0:K]  # M channels are chosen to be accessed               
            for i in range(len(arm_access_chosen)):
                Reward_all_simulation_mc[m_c][t] += np.dot(feature_generation.T, theta_generation[arm_access_chosen[i]])                 
        '''
        # ---------------------------------------------------------- #
        '''
        # Softmax algorithm, we ignore this algo in this file 
        # if t < T_init:
        #     for i, theta_index in enumerate(arm_theta_set):
        #         estimated_reward_greedy[i] = feature_generation[0]*np.random.binomial(1,theta_generation[i][0],1) + feature_generation[1]*theta_generation[i][1] + feature_generation[2] * np.random.normal(theta_generation[i][2],0.1) + feature_generation[3] * theta_generation[i][3] + feature_generation[4] * theta_generation[i][4] 
        #         Softmax_algo.update( i, estimated_reward_greedy[i] )
        # else:
        #     for i in range(Channel_Num):
        #         reward[i] = feature_generation[0]*np.random.binomial(1,theta_generation[i][0],1) + feature_generation[1]*theta_generation[i][1] + feature_generation[2] * np.random.normal(theta_generation[i][2],0.1) + feature_generation[3] * theta_generation[i][3] + feature_generation[4] * theta_generation[i][4] 
        '''
        # ---------------------------------------------------------- #
        
        # the performance benchmark, we find channels with optimal utility values 
        Reward = []
        for i, theta_index in enumerate(arm_theta_set):
            Reward.append( np.dot(feature_generation.T, theta_generation[i])  ) 
        Reward_optimal_mc[m_c, t] = sum(sorted(Reward, reverse = True)[0:M])
       
# In[8] plot the figure 

t_x = list(range(T_simulation))
reward_difference_mc = Reward_optimal_mc - Reward_all_simulation_mc

regret_x_label = [ round(0.001 * T_simulation), round(0.005 * T_simulation), round(0.01 * T_simulation),   round(0.03 * T_simulation), round(0.05 * T_simulation), round(0.1 * T_simulation), round(0.2 * T_simulation), round(0.3 *T_simulation), round(0.4 *T_simulation), round(0.5 *T_simulation), round(0.6 *T_simulation), round(0.7 *T_simulation), round(0.8 *T_simulation), round(0.9 *T_simulation),  T_simulation  ]


cumulative_regret_y_label   = np.zeros(len(regret_x_label))
accumulative_reward_y_label = np.zeros(len(regret_x_label))


for i,element in enumerate(regret_x_label):
    for j in range(reward_difference_mc.shape[0]):
        cumulative_regret_y_label[i] += sum(reward_difference_mc[j,0:element])
        accumulative_reward_y_label[i] += sum(Reward_all_simulation_mc[j][0:element])
    cumulative_regret_y_label[i] = cumulative_regret_y_label[i]/(m_c_number  )
    accumulative_reward_y_label[i] = accumulative_reward_y_label[i]/(m_c_number*element)
    
plt.plot(regret_x_label, cumulative_regret_y_label,'-^')
plt.title('regret and time slots')
plt.show()
 

chosen_time_matrix = []
for n in range(len_Num):
    chosen_time_matrix.append( policy[n].t_count )

Throuhgput = sum(Throughput_mc)/m_c_number
Price = sum(Price_cost_mc)/m_c_number

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


# we finally record the following results in 'plot_algorithm_single)user_price_QoS.py'
print('The regret is', cumulative_regret_y_label)
print('The average_throughput is', average_throughput)
print('The average_cost is', average_cost)

plt.plot(regret_x_label, average_throughput,'-^')
plt.title('average throughput and time slots')
plt.show()

plt.plot(regret_x_label, average_cost,'-^')
plt.title('average cost and time slots')
plt.show()


# we record the 'cumulative_regret_y_label', 'average_throughput', and 'average_cost' in 'plot_algorithm_single_user_price_QoS.py'


