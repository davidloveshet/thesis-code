# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 08:41:00 2021

@author: xxx

M = K = 2
The optimal channel allocation is:
    user 1: 1,8
    user 2: 2,3
    user 3: 7,4

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
import Auctioning as Auction
import random 
from random import shuffle

dimension = 2
feature = np.zeros(dimension).reshape(dimension,1)
lambdada = 1
alpha = 1 # the algorithm parameter

# T_simulation = 16000 # the overall simulation time
# m_c_number = 10

T_simulation = 16000 # the overall simulation time
m_c_number = 5 # Monte Carlo Number  
T_init = 1           # the initial verification time
ALGO = 'proposed'  # 'random', 'fixed', 'proposed'. select various options, we can obtain the corresponding algorithm 


# In[11]
U_Num = 3
M = 2
U = U_Num * M 
 

# In[0] Bertsekas auction algorithm 
def create_matrices(number_of_agents):
    """
    Create matrices of agents and their values for varying objects
    """
    matrix = []
    agent = 0
    while agent != number_of_agents:
        temp_matrix = []
        for prices in range(0,number_of_agents):
            temp_matrix.append(round(random.uniform(1, 15),0))
        matrix.append(temp_matrix)
        agent += 1
    prices_list = []
    for prices in range(0,number_of_agents):
        prices_list.append(round(random.uniform(1, 10),0))
    return matrix, prices_list

def assign_values(number_of_agents):
    """
    Assign the values to each of the agents
    """
    temp_list = []
    agents = []
    for i in range(0,number_of_agents):
        temp_list.append(i)
    shuffle(temp_list)
    for p in range(0,number_of_agents):
        agents.append([temp_list[p],0])
    return agents

def get_best_option(agent_number, matrix, costs):
    """
    Find the best option for the given agent
    """
    diff_list = []
    for i in range(0,len(costs)):
        diff_list.append(matrix[agent_number][i] - costs[i])
    highest_val = max(diff_list)
    ind_highest_val = diff_list.index(highest_val)
    diff_list[ind_highest_val] = -1000000000
    second_highest_val = max(diff_list)
    ind_second_highest_val = diff_list.index(max(diff_list))
    return highest_val, ind_highest_val, second_highest_val, ind_second_highest_val

def find_index_of_val(mylist, value):
    """
    Find index of given value in list
    """
    l = [i[0] for i in mylist]
    if value in l:
        return l.index(value) 
    else:
        return []
    # 如果有值，则进行交换，如果没有值，则进行满意

def check_happiness(agent_matrix,payoff_matrix,cost_list):
    """
    Check if agents are happy with their current situation
    """
    for i in range(0,len(agent_matrix)):
        if agent_matrix[i][1] == 0:
            high,ind_high,sec_high,sec_ind_high = get_best_option(i,payoff_matrix,cost_list)
            if ind_high == agent_matrix[i][0]:
                agent_matrix[i][1] = 1

def sophisticated_auction(epsilon, agents, payoff_matrix, cost_list):
    """
    Running the sophisticated version of the auction algorithm where we force agents to increase bids by epsilon each time they bid
    """
    item_Num = 0
    while sum(n for _, n in agents) != len(agents):
        for i in range(0,len(agents)):
            check_happiness(agents, payoff_matrix, cost_list)
            item_Num = item_Num + 1
            if agents[i][1] == 0:
                high,ind_high,sec_high,sec_ind_high = get_best_option(i, payoff_matrix, cost_list)
                switch_index = find_index_of_val( agents, int(ind_high) )
                if switch_index == []: # 如果为 switch_index 没有用户占用，则不需要交换
                    agents[i][1] = 1
                    agents[i][0] = ind_high
                else:
                    agents[switch_index][1] = 0
                    agents[i][1] = 1 
                    agents[switch_index][0] = agents[i][0]
                    agents[i][0] = ind_high
                    cost_list[ind_high] = cost_list[ind_high] + abs(((payoff_matrix[i][ind_high] - cost_list[ind_high]) - (payoff_matrix[i][sec_ind_high] - cost_list[sec_ind_high]))) + epsilon
    return item_Num
# '''

def Indicator_func(y,x):
    if y == x:
        return 1
    else:
        return 0
    
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

len_Num = len(arm_theta_set_SU_1)
arm_theta_set_SU_1 = arm_theta_set_SU_1
arm_theta_set_SU_1 = [arm_theta_set_SU_1[i].reshape(dimension,1) for i in range(len_Num)]
 
arm_theta_set = arm_theta_set_SU_1
theta_generation_1 = arm_theta_set_SU_1
Channel_Num = len(arm_theta_set)

# In[1] define the channel set for SU 2
arm_theta_set_SU_2 = [
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

arm_theta_set_SU_2 = [ arm_theta_set_SU_2[i].reshape(dimension,1) for i in range(9) ]
theta_generation_2 = arm_theta_set_SU_2

# In[1] define the channel set for SU 3
arm_theta_set_SU_3 = [
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

arm_theta_set_SU_3 = [ arm_theta_set_SU_3[i].reshape(dimension,1) for i in range(9) ]
theta_generation_3 = arm_theta_set_SU_3

 
# In[0]
 
feature_generation_1 = np.array([  [1], [0]  ])
feature_generation_2 = np.array([  [0.9701], [0.2425] ])
feature_generation_3 = np.array([  [0.7071], [0.7071] ])


reward_1_temp = []
reward_1_1_temp = []
reward_2_temp = []
reward_2_1_temp = []
reward_3_temp = []
reward_3_1_temp = []

for i in range(len(arm_theta_set_SU_3)):
    reward_1_temp.append(np.dot(feature_generation_1.T, arm_theta_set_SU_1[i]).tolist()[0][0])
    reward_1_1_temp.append(np.dot(feature_generation_1.T, arm_theta_set_SU_1[i]).tolist()[0][0])
    reward_2_temp.append(np.dot(feature_generation_2.T, arm_theta_set_SU_2[i]).tolist()[0][0])
    reward_2_1_temp.append(np.dot(feature_generation_2.T, arm_theta_set_SU_2[i]).tolist()[0][0])
    reward_3_temp.append(np.dot(feature_generation_3.T, arm_theta_set_SU_3[i]).tolist()[0][0])
    reward_3_1_temp.append(np.dot(feature_generation_3.T, arm_theta_set_SU_3[i]).tolist()[0][0])


# Reward_Matrix = [ 
# [0.670265, 0.465147, 0.675702, 0.449578,0.341967, 0.750437, 0.65507, 0.853193, 0.789157],
# [0.670265, 0.465147, 0.675702, 0.449578,0.341967, 0.750437, 0.65507, 0.853193, 0.789157],
# [0.670265, 0.465147, 0.675702, 0.449578,0.341967, 0.750437, 0.65507, 0.853193, 0.789157]
# ]

Reward_Matrix = [
reward_1_temp,
reward_1_1_temp,
reward_2_temp,
reward_2_1_temp,
reward_3_temp,
reward_3_1_temp ]
       
        
profit_matrix = Reward_Matrix
hungarian = Hungarian()

hungarian.calculate(profit_matrix, is_profit_matrix=True)
print("the overal revenue is", hungarian.get_total_potential() )
print("Results:\n\t", hungarian.get_results())
optimal_result = hungarian.get_results()
optimal_revenue = hungarian.get_total_potential()
print("-" * 80)  



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



Reward_all_simulation = np.zeros(T_simulation)
Reward_optimal = np.zeros(T_simulation)


# Benchmark 
Reward_optimal_mc = np.ones((m_c_number, T_simulation)) * optimal_revenue

# In[6] Simulate more channels at one slot, there are N channels, M channels each slot

Reward = []

Reward_all_simulation_mc = np.zeros((m_c_number, T_simulation))
Reward_all_simulation_mc_SU_1 = np.zeros((m_c_number, T_simulation))
Reward_all_simulation_mc_SU_2 = np.zeros((m_c_number, T_simulation))
Reward_all_simulation_mc_SU_3 = np.zeros((m_c_number, T_simulation))
Throughput_mc_1 = np.zeros((m_c_number, T_simulation))
Price_cost_mc_1 = np.zeros((m_c_number, T_simulation))

Throughput_mc_2 = np.zeros((m_c_number, T_simulation))
Price_cost_mc_2 = np.zeros((m_c_number, T_simulation))

Throughput_mc_3 = np.zeros((m_c_number, T_simulation))
Price_cost_mc_3 = np.zeros((m_c_number, T_simulation))

reward_difference_mc = np.zeros((m_c_number, T_simulation))
epsilon_bertsekas = 0.01

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
                reward_SU_1[i] = feature_generation_1[0]*np.random.binomial(1,theta_generation_1[i][0],1) + feature_generation_1[1]*theta_generation_1[i][1]  
                hat_theta_SU_1[i] = policy_SU_1[i].update_information(feature_generation_1, reward_SU_1[i])
                ucb_s_SU_1[i] = policy_SU_1[i].compute_index(hat_theta_SU_1[i], feature_generation_1, policy_SU_1[i].invcov, t+1)
                
                reward_SU_2[i] = feature_generation_2[0]*np.random.binomial(1,theta_generation_2[i][0],1) + feature_generation_2[1]*theta_generation_2[i][1]  
                hat_theta_SU_2[i] = policy_SU_2[i].update_information(feature_generation_2, reward_SU_2[i])
                ucb_s_SU_2[i] = policy_SU_2[i].compute_index(hat_theta_SU_2[i], feature_generation_2, policy_SU_2[i].invcov, t+1)                
                
                reward_SU_3[i] = feature_generation_3[0]*np.random.binomial(1,theta_generation_3[i][0],1) + feature_generation_3[1]*theta_generation_3[i][1]  
                hat_theta_SU_3[i] = policy_SU_3[i].update_information(feature_generation_3, reward_SU_3[i])
                ucb_s_SU_3[i] = policy_SU_3[i].compute_index(hat_theta_SU_3[i], feature_generation_3, policy_SU_3[i].invcov, t+1)   
                
        else:
            Reward_Matrix_estimated = [ ucb_s_SU_1, ucb_s_SU_1, ucb_s_SU_2, ucb_s_SU_2, ucb_s_SU_3, ucb_s_SU_3 ]
            # if M_U == 2:
            #     Reward_Matrix_estimated = [ ucb_s_SU_1, ucb_s_SU_1, ucb_s_SU_2, ucb_s_SU_2, ucb_s_SU_3, ucb_s_SU_3 ]
            # order the channel ucb_s to output the M channels, which used for accessing selection
            # mixer = np.random.random(ucb_s.size)
            # ucb_indices = list(np.lexsort((mixer, ucb_s))) 
            # output = ucb_indices[::-1]
            # chosen_arm = output[0:M]  # M channels are chosen to be sensed 
            
            if ALGO == 'random':
                Reward_Matrix_estimated = np.random.random((U,Channel_Num))            
            
            """
            Hungarian model
            """            
            '''
            # calculate the allocation relationship using Hungarian on Reward_Matrix_estimated
            hungarian.calculate(Reward_Matrix_estimated, is_profit_matrix=True)
                # print("the overal revenue is", hungarian.get_total_potential() )
            # if t % 200 == 0:
            #     print("The time is", t )
            #     print("Results:\n\t", hungarian.get_results())
            optimal_result = hungarian.get_results() # we obtain the whole allocation results
            optimal_revenue = hungarian.get_total_potential()            
            '''
     
            """
            添加 Bertsekas 模块，注意我们在 auction 算法中的 allocation_relationship 和 hungarian 算法中的 allocation_relationship 交换了一下
            """
            ''' origin auction 
            agents = assign_values(U)
            cost_list = np.zeros(len(arm_theta_set_SU_1)).tolist()
            Iter_Num = 0
            Iter_Num = sophisticated_auction(epsilon_bertsekas, agents, Reward_Matrix_estimated, cost_list)
            allocation_relationship_origin = []
            for u in range(U):
                allocation_relationship_origin.append( [ u, agents[u][0] ])  
            allocation_relationship = copy.deepcopy(allocation_relationship_origin)
            optimal_result = allocation_relationship
            '''
            
            if ALGO == 'fixed':
                # optimal_result = [ (0, 0), (1, 7), (2, 1), (3, 2), (4, 6), (5, 3)]
                optimal_result = [ [0,7], [1,8], [2,5], [3,6], [4,3], [5,4] ]
    
            for i in range(Channel_Num):
                channel_state_1[i] = np.random.binomial(1,theta_generation_1[i][0],1)
                channel_state_2[i] = np.random.binomial(1,theta_generation_2[i][0],1)
                channel_state_3[i] = np.random.binomial(1,theta_generation_3[i][0],1)
                reward_SU_1[i] = feature_generation_1[0]*channel_state_1[i] + feature_generation_1[1]*theta_generation_1[i][1]  
                reward_SU_2[i] = feature_generation_2[0]*channel_state_2[i] + feature_generation_2[1]*theta_generation_2[i][1]  
                reward_SU_3[i] = feature_generation_3[0]*channel_state_3[i] + feature_generation_3[1]*theta_generation_3[i][1]   
            for j in range(len(optimal_result)):
                if optimal_result[j][0] == 0 or optimal_result[j][0] == 1: # SU 1, accesses the [j][1] channel
                    channel_access = optimal_result[j][1]
                    hat_theta_SU_1[channel_access] = policy_SU_1[channel_access].update_information(feature_generation_1, reward_SU_1[channel_access])
                    policy_SU_1[channel_access].count_the_sensed_time()
                    Reward_all_simulation_mc_SU_1[m_c][t] += np.dot(feature_generation_1.T, theta_generation_1[channel_access])
                    if channel_state_1[channel_access] == 1:
                        Throughput_mc_1[m_c][t] += channel_state_1[channel_access] # theta_generation_1[channel_access][0]
                        Price_cost_mc_1[m_c][t] += theta_generation_1[channel_access][1]
                elif optimal_result[j][0] == 2 or optimal_result[j][0] == 3: # SU 2, accesses the [j][1] channel
                    channel_access = optimal_result[j][1]
                    hat_theta_SU_2[channel_access] = policy_SU_2[channel_access].update_information(feature_generation_2, reward_SU_2[channel_access])
                    policy_SU_2[channel_access].count_the_sensed_time()
                    Reward_all_simulation_mc_SU_2[m_c][t] += np.dot(feature_generation_2.T, theta_generation_2[channel_access])
                    if channel_state_2[channel_access] == 1:
                        Throughput_mc_2[m_c][t] += channel_state_2[channel_access] # theta_generation_2[channel_access][0]
                        Price_cost_mc_2[m_c][t] += theta_generation_2[channel_access][1]
                elif optimal_result[j][0] == 4 or optimal_result[j][0] == 5:
                    channel_access = optimal_result[j][1]
                    hat_theta_SU_3[channel_access] = policy_SU_3[channel_access].update_information(feature_generation_3, reward_SU_3[channel_access])
                    policy_SU_3[channel_access].count_the_sensed_time()
                    Reward_all_simulation_mc_SU_3[m_c][t] += np.dot(feature_generation_3.T, theta_generation_3[channel_access])
                    if channel_state_3[channel_access] == 1:
                        Throughput_mc_3[m_c][t] += channel_state_3[channel_access] # theta_generation_3[channel_access][0]
                        Price_cost_mc_3[m_c][t] += theta_generation_3[channel_access][1]                
    
            for element in range(Channel_Num):    
                ucb_s_SU_1[element] = policy_SU_1[element].compute_index(hat_theta_SU_1[element], feature_generation_1, policy_SU_1[element].invcov, t+1)
                ucb_s_SU_2[element] = policy_SU_2[element].compute_index(hat_theta_SU_2[element], feature_generation_2, policy_SU_2[element].invcov, t+1)
                ucb_s_SU_3[element] = policy_SU_3[element].compute_index(hat_theta_SU_3[element], feature_generation_3, policy_SU_3[element].invcov, t+1)
  
# In[8] plot the figure 

t_x = list(range(T_simulation))
Reward_all_simulation_mc = Reward_all_simulation_mc_SU_1 + Reward_all_simulation_mc_SU_2 + Reward_all_simulation_mc_SU_3
reward_difference_mc = Reward_optimal_mc - Reward_all_simulation_mc

regret_x_label = [ round(0.005 * T_simulation), round(0.01 * T_simulation), round(0.03 * T_simulation), round(0.05 * T_simulation),  round(0.07 * T_simulation), round(0.1 * T_simulation), round(0.2 * T_simulation), round(0.3 *T_simulation), round(0.4 *T_simulation), round(0.5 *T_simulation), round(0.6 *T_simulation), round(0.7 *T_simulation), round(0.8 *T_simulation), round(0.9 *T_simulation),  T_simulation  ]


cumulative_regret_y_label   = np.zeros(len(regret_x_label))
accumulative_reward_y_label = np.zeros(len(regret_x_label))


for i,element in enumerate(regret_x_label):
    for j in range(reward_difference_mc.shape[0]):
        cumulative_regret_y_label[i] += sum(reward_difference_mc[j,0:element])
        accumulative_reward_y_label[i] += sum(Reward_all_simulation_mc[j][0:element])
    cumulative_regret_y_label[i] = cumulative_regret_y_label[i]/(m_c_number  )
    accumulative_reward_y_label[i] = accumulative_reward_y_label[i]/(m_c_number*element)
    
plt.plot(regret_x_label, cumulative_regret_y_label,'b-^')
plt.title('regret and time slots')
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
plt.title('price and time slots')
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
plt.title('average throughput and time slots')
plt.show()


# we record the 'cumulative_regret_y_label' (regret), 'average_throughput_1', 'average_throughput_2', 'average_throughput_3' for each user,
# record 'average_Price_1', 'average_Price_2', 'average_Price_3' for each user in 'plot_algorithm_price_QoS_Central.py'










