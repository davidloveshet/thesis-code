# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 12:13:23 2021

该代码用于求得最优累积期望吞吐量和最优能量效率

@author: xxx

1. 在该仿真中， 累积吞吐量的benchmark，reward_optimal_value_benchmark 可通过设置较大的能量budget，固定 M 条信道获取
2. M 表示每时隙中信道感知的数量，可通过选取不同值表示不同感知能力，在文中 M=1,2,3,4。Monte_Carlo_Num 表示蒙特卡洛次数
"""

# 备注：求拉动次数的时候，需要设置为 reward 和 cost 随机值


import numpy as np
import random 
import matplotlib.pyplot as plt
import time
import math
import copy
# 生成随机数，然后求和，然后除法，得到向量

start = time.time() # 记录时间


 
# In[0] 超参数设置
alpha_0 = 1/100
alpha_1 = 1/2

# In[1] 参数设置
# budget_list = [   0.3, 0.8, 1, 3 , 8 , 10 , 20, 30, 40 ,50, 60, 70   ]  # 单位为 J，能量大小
budget_list = [  20    ]  # 单位为 J，能量大小
channel_idle_prob = [ 0.972, 0.824, 0.742, 0.644, 0.528, 0.462, 0.341, 0.225, 0.109, 0.045 ] # 随机生成存储
channel_noise_set = [ -104, -103, -102, -101, -100, -99, -98, -97, -96, -95 ] # 噪声功率 dBm
bandwidth = 1 # MHz
varepsilon_threshold = 0 # 设置次要用户能量阈值
transmission_rate_required = 2 # Mbps 
transmission_duration = 90e-03 # s
transmission_distance = 150 # m
Tx_Num = 4 # 发射天线数量
K_factor = 3 #莱斯因子


# In[0] 可改变参数
M = 3 # 每时隙中次要用户可感知信道数量
Monte_Carlo_Num = 30 # 蒙特卡洛次数



# In[1] 定义class
class algo:
    def __init__(self, channel_info):
        self.channel_info = channel_info
 
        
        # some other parameters
        self.estimated_theta = 0
        self.accumulative_reward = 0
        self.accumulative_cost = 0
        self.mean_reward = 0
        self.mean_cost = 0
        self.chosen_time = 0
        self.index = 0
        
    def compute_index(self, instant_reward, time, instant_cost):
        self.accumulative_reward = self.accumulative_reward + instant_reward
        self.accumulative_cost = self.accumulative_cost + instant_cost
        self.chosen_time = self.chosen_time + 1
        self.mean_reward = (self.accumulative_reward)/(self.chosen_time)
        self.mean_cost = (self.accumulative_cost)/(self.chosen_time)
        
        # self.index = self.mean_reward/self.mean_cost + np.sqrt(np.log(time) / (self.chosen_time))
        # self.index_classic_MAB = self.mean_reward + np.sqrt(np.log(time) / (self.chosen_time))
        # self.D_ONES = self.mean_reward + np.sqrt(np.log(time) / (self.chosen_time)) - (   self.cost - 1.5 * np.sqrt(np.log(time) / (self.chosen_time))  )
        # self.ONES = self.mean_reward - self.cost * ( self.mean_reward/self.cost - np.sqrt(np.log(time) / (self.chosen_time))  )+  np.sqrt(np.log(time) / (self.chosen_time))
        # return self.index

    
    def compute_index_update(self, instant_reward, time):
        self.index = alpha_0 * self.mean_reward/(self.mean_cost) +  np.sqrt( alpha_1 * np.log(time) / (self.chosen_time))
        return self.index
        
    def initial(self):
        self.estimated_theta = 0
        self.accumulative_reward = 0
        self.mean_reward = 0
        self.chosen_time = 0
        self.index = 0
        self.index_classic_MAB = 0
        self.accumulative_cost = 0
        self.mean_cost = 0        
        
class channel_info:
    def __init__(self, c_s):
        self.c_s = c_s
 
 

def random_pick(some_list, probabilities): 
    x = random.uniform(0,1) 
    cumulative_probability = 0.0 
    for item, item_probability in zip(some_list, probabilities): 
         cumulative_probability += item_probability 
         if x < cumulative_probability:
               break 
    return item                   

# In[2] 感知信道的能耗，量纲为 J 
c_1_s = 0.55e-03
c_2_s = 0.55e-03
c_3_s = 0.55e-03
c_4_s = 0.55e-03
c_5_s = 0.55e-03
c_6_s = 0.55e-03
c_7_s = 0.55e-03
c_8_s = 0.55e-03
c_9_s = 0.55e-03
c_10_s = 0.55e-03

 

# In[3] Channel_gain_noise_ratio, Tx_Num 表示发射天线数量
# 瑞丽衰落
# def Rayleigh_pathloss_function(transmission_distance, Tx_Num, channel_noise):
#     # channel noise is in the form of num
#     PathLoss_Model_Parameter = np.sqrt( 10**(-3.53)/( transmission_distance**(3.76)) ) # 路损模型 
#     Channel_parameter_vector = []
#     for i in range( Tx_Num ):
#         Channel_parameter_vector.append( PathLoss_Model_Parameter * 1/np.sqrt(2) * ( np.random.normal() + complex(0, 1) * np.random.normal() ) )
#         # 其他场景直接求信噪比比较好 或者信道系数
#     Channel_parameter_norm = np.linalg.norm( Channel_parameter_vector )
#     channel_noise = 10**(channel_noise/10) * 10**(-3)
#     Channel_gain_noise_ratio = Channel_parameter_norm**2/channel_noise
#     return Channel_gain_noise_ratio

# 莱斯衰落
def Rician_pathloss_function(transmission_distance, Tx_Num, channel_noise, K_factor):
    # channel noise is in the form of num
    PathLoss_Model_Parameter = np.sqrt( 10**(-3.53)/( transmission_distance**(3.76)) )
    Channel_parameter_vector = []
    for i in range( Tx_Num ):
        Channel_parameter_vector.append( PathLoss_Model_Parameter * ( np.sqrt( K_factor/(K_factor+1) ) * ( np.random.normal() + complex(0, 1) * np.random.normal() ) + np.sqrt( 1/(K_factor+1) ) * np.random.normal()   )  )
        # 其他场景直接求信噪比比较好 或者信道系数
    Channel_parameter_norm = np.linalg.norm( Channel_parameter_vector )
    channel_noise = 10**(channel_noise/10) * 10**(-3)
    Channel_gain_noise_ratio = Channel_parameter_norm**2/channel_noise
    return Channel_gain_noise_ratio


 

# 所需能量
def required_energy(bandwidth, Rate, channel_gain_noise_ratio, transmission_duration):
    required_power = (2**(Rate/bandwidth)-1)/channel_gain_noise_ratio
    required_energy = required_power * transmission_duration
    return required_energy


 
# In[4] 每条信道定义一个 class
n_1 = channel_info(c_1_s)
n_2 = channel_info(c_2_s)
n_3 = channel_info(c_3_s)
n_4 = channel_info(c_4_s)
n_5 = channel_info(c_5_s)
n_6 = channel_info(c_6_s)
n_7 = channel_info(c_7_s)
n_8 = channel_info(c_8_s)
n_9 = channel_info(c_9_s)
n_10 = channel_info(c_10_s)

channel_set = [ n_1, n_2, n_3, n_4, n_5, n_6, n_7, n_8, n_9, n_10 ]


N = len(channel_set) # 信道数量
channel_state_set = np.zeros(N)
Channel_gain_noise_ratio_set = np.zeros(N) 


# In[0] 求最优信道集合，选择 {\mu_i^r}/{\mu_i^c} 最优的 M 条信道 
# \mu_i^r 表示空闲概率最大的信道，由于信道质量分布相同 \mu_i^c 与信道噪声有关

channel_optimality_order = np.zeros(N) 
for i in range(N):
    channel_optimality_order[i] = copy.deepcopy( channel_idle_prob[i]/10**(channel_noise_set[i]/10) * 10**(-3) )
    
mixer = np.random.random(channel_optimality_order.size)
channel_optimality_order_indices = list(np.lexsort((mixer, channel_optimality_order))) 
output = channel_optimality_order_indices[::-1]
chosen_arm_optimality = output[0:M]       
    

# In[1] 设计算法的仿真 
    
policy = [] # 每条信道信息更新过程定义一个class
for i in range(N):
    policy.append(algo(channel_set[i]))


sum_reward_set = np.zeros(len(budget_list)) 
reward_optimal_value_benchmark = np.zeros(len(budget_list)) 
average_reward_set = np.zeros(len(budget_list)) # 能量效率
reward_difference_set = np.zeros(len(budget_list))  
normalized_reward_difference_set = np.zeros(len(budget_list))
 

for index_, budget in enumerate(budget_list):
    
    simulation_Num_ = math.ceil(budget * 1000)
    reward_monte_set = np.zeros( (Monte_Carlo_Num, simulation_Num_) )
    reward_monte_mean = np.zeros(simulation_Num_)
    end_time_budget = np.zeros(Monte_Carlo_Num)    
    
    sum_reward = np.zeros(Monte_Carlo_Num)
    reward_difference = np.zeros(Monte_Carlo_Num)
    normalized_reward_difference = np.zeros(Monte_Carlo_Num)
    a_Num_chosen = np.zeros((1,N)).tolist()[0]
    print("The budget is", budget)
    
    for monte in range(Monte_Carlo_Num):
        print("This is the", monte, "-th simulation")
        for i in range(len(policy)):
            policy[i].initial()
    
        reward = np.zeros(N)
        cost_ = np.zeros(N) 
        index = np.zeros(N)
        chosen_arm = []
        
        reward_monte = np.zeros(simulation_Num_)
        
        budget_residual = budget
    
        for num in range(simulation_Num_):
            

            
            # 实时信噪比、信道状态、接入信道获得的吞吐量、能量消耗
            for i in range(N):
                Channel_gain_noise_ratio_set[i] = Rician_pathloss_function( transmission_distance, Tx_Num, channel_noise_set[i], K_factor)
                channel_state_set[i] = np.random.binomial(1,channel_idle_prob[i],1)
                reward[i] = transmission_rate_required * transmission_duration * channel_state_set[i] # 信道 i 的可达速率
                cost_[i] = channel_set[i].c_s + required_energy(bandwidth, transmission_rate_required, Channel_gain_noise_ratio_set[i], transmission_duration) * channel_state_set[i] # 在信道 i 上传输所需能量
            
 
            temp_reward = 0
            temp_cost = 0
            
            if num < 1: # 初始化相关信息，没有 reward 和 能耗 
                pass 
            else:

                # ''' 说明：求 benchmark 时采用固定的 chosen_arm
                chosen_arm = copy.deepcopy( chosen_arm_optimality )
                # '''
 
                for i, element in enumerate(chosen_arm):
                    policy[element].compute_index(reward[element], num, cost_[element])   # 对选择的信道相关信息进行更新
                    temp_reward += reward[element]
                    temp_cost += cost_[element]
                    
                budget_residual = budget_residual - temp_cost
                reward_monte[num] = temp_reward
            if budget_residual < varepsilon_threshold: # 自己设置的剩余能量阈值
                break
        end_time_budget[monte] = num
 
           
        reward_monte_set[monte] = reward_monte
        
    reward_monte_mean = sum(reward_monte_set)/Monte_Carlo_Num
    
    sum_reward_set[index_] = sum(reward_monte_mean)
    average_reward_set[index_] = sum_reward_set[index_]/budget # 平均能量效率

 
    
 

 
# In[16] 

plt.plot(budget_list, average_reward_set) # 平均能量效率

# plt.plot(budget_list, reward_difference_set) # 吞吐量损失

print("The optimal average reward efficiency is", average_reward_set)




end = time.time() # 记录结束时间
print("程序运行时间: " + str(end-start) + "秒")












