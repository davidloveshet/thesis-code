# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 12:13:23 2021
Modified on May 7 2023

@author: xxx
PS: This file is to simulate the energy-constraint algorithm for the single user 
离散概率场景是参考IEEE Fellow Qing Zhao 的论文
说明：
1. 单个次要用户能量有限下的感知接入
2. 参数 M 为次要用户信道感知数量，在该仿真中 M = 1,2,3,4
"""

# In[0] 
import numpy as np
import random 
import matplotlib.pyplot as plt
import time
import math
import copy
# 生成随机数，然后求和，然后除法，得到向量

start = time.time() # 记录时间

def Indicator_func(y,x):
    if y == x:
        return 1
    else:
        return 0
    
# In[1] 参数设置
alpha_0 = 2
alpha_1 = 1/2
Transmission_Rate = 1 # Mbps
Transmission_Duration = 18e-03  # s
M = 4                # 信道感知数量
budget_list = [  1000, 3000, 8000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110e03, 120e03, 130e03, 140e03, 150e03, 160e03    ] # 次要用户能量限制大小，注意的是其单位为 10^{-3} J，因为感知、接入能量消耗量级皆为 10^{-3} J# 实际上上述 budget 为 [1,3,8,10,30,40,50,60,70,80,90,100,110,120]   J
 
Monte_Carlo_Num = 200  # 蒙特卡洛仿真次数
varepsilon_threshold = 0 # 设置次要用户能量阈值

# In[2] 定义概率函数，some_list = [0,1,2,3], probabilities = [0.1,0.2,0.7]，则列表中每个元素以对应的概率出现
def random_pick(some_list, probabilities): 
    x = random.uniform(0,1) 
    cumulative_probability = 0.0 
    for item, item_probability in zip(some_list, probabilities): 
         cumulative_probability += item_probability 
         if x < cumulative_probability:
               break 
    return item 

# In[3] 定义算法class，每条信道的相关信息更新过程在该class中记录
class algo:
    def __init__(self, channel_info):
        self.channel_info = channel_info
        self.cost = self.channel_info.cost 
        
        # some other parameters
        self.estimated_theta = 0
        self.accumulative_reward = 0
        self.accumulative_cost = 0
        self.mean_reward = 0
        self.mean_cost = 0
        self.chosen_time = 0
        self.index = 0
        
    # 更新信道相关信息
    def compute_index(self, instant_reward, time, instant_cost):
        self.accumulative_reward = self.accumulative_reward + instant_reward
        self.accumulative_cost = self.accumulative_cost + instant_cost
        self.chosen_time = self.chosen_time + 1
        self.mean_reward = (self.accumulative_reward)/(self.chosen_time) # 平均能耗，单位为 M bits 
        self.mean_cost = (self.accumulative_cost)/(self.chosen_time)
        
        # self.index = self.mean_reward/self.mean_cost + np.sqrt(np.log(time) / (self.chosen_time))
        # self.index_classic_MAB = self.mean_reward + np.sqrt(np.log(time) / (self.chosen_time))
        # self.D_ONES = self.mean_reward + np.sqrt(np.log(time) / (self.chosen_time)) - (   self.cost - 1.5 * np.sqrt(np.log(time) / (self.chosen_time))  )
        # self.ONES = self.mean_reward - self.cost * ( self.mean_reward/self.cost - np.sqrt(np.log(time) / (self.chosen_time))  )+  np.sqrt(np.log(time) / (self.chosen_time))
        # return self.index

    # 每时隙中用于更新信道标号
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

# In[4] 为每条信道相关信息定义并存储在class中
class channel_info:
    def __init__(self, p, cost, theta, c_s, c_a):
        self.cost = cost
        self.p = p
        self.c_s = c_s
        self.c_a = c_a
        self.theta = theta
        # self.mu = self.theta
        # self.mu = Transmission_Rate * Transmission_Duration * self.theta

        # 每时隙消耗的能量写在后续程序中，此处略过
    # def cost_random(self):
    #     self.cost_random_ = self.c_s + np.random.binomial(1,self.theta,1) * random_pick( self.c_a, self.p)
    #     return self.cost_random_
 
# In[5] 感知决策和信道感知过程消耗的能量，数据来源于 Ying-Chang Liang 老师发表于 IEEE Journal on Selected Areas in Communications 中 《Energy-Efficient Design of Sequential Channel
# Sensing in Cognitive Radio Networks: Optimal Sensing Strategy, Power Allocation, and Sensing Order》 仿真的数据。次要用户信道感知功率为 110 mW，感知时间设置为 5 ms，因此信道感知耗能为 550 e-06 = 0.55 e-03 J。
# 在本仿真过程中 能耗量纲为 1e-03J，因此 550e-06 (J) 为 0.55 ( 10^{-3} J )

c_1_s = 0.55  
c_2_s = 0.55 
c_3_s = 0.55 
c_4_s = 0.55  
c_5_s = 0.55  
c_6_s = 0.55  
c_7_s = 0.55  
c_8_s = 0.55  
c_9_s = 0.55  
c_10_s = 0.55  
c_11_s = 0.55  
c_12_s = 0.55  
 
# In[6] 信道接入过程中消耗的能量，能量消耗量纲皆为 1e-03J， 1.8e-03 表示的最大传输耗能，0.1e-03 表示的是探测+反馈ACK的耗能 （仿真中略去了 1e-03 的量纲）。数据来源于Ying-Chang Liang 老师发表于 IEEE Journal on Selected Areas in Communications 中 《Energy-Efficient Design of Sequential Channel
# Sensing in Cognitive Radio Networks: Optimal Sensing Strategy, Power Allocation, and Sensing Order》 仿真的数据。文献中次要用户发射功率最大设置为 166.62 mW，在该仿真中设置最大发射功率为 100 mW，每时隙传输的时长设置为 18 ms，因此 传输中消耗的能量为 1800 e-06 J = 1.8 e-03 J。

c_1_a = [ 1/4 * 1.8  + 0.1 , 1/2 * 1.8  + 0.1 , 3/4 * 1.8  + 0.1 , 1.8  + 0.1  ]
c_2_a = [ 1/4 * 1.8  + 0.1 , 1/2 * 1.8  + 0.1 , 3/4 * 1.8  + 0.1 , 1.8  + 0.1  ] 
c_3_a = [ 1/4 * 1.8  + 0.1 , 1/2 * 1.8  + 0.1 , 3/4 * 1.8  + 0.1 , 1.8  + 0.1  ] 
c_4_a = [ 1/4 * 1.8  + 0.1 , 1/2 * 1.8  + 0.1 , 3/4 * 1.8  + 0.1 , 1.8  + 0.1  ] 
c_5_a = [ 1/4 * 1.8  + 0.1 , 1/2 * 1.8  + 0.1 , 3/4 * 1.8  + 0.1 , 1.8  + 0.1  ] 
c_6_a = [ 1/4 * 1.8  + 0.1 , 1/2 * 1.8  + 0.1 , 3/4 * 1.8  + 0.1 , 1.8  + 0.1  ] 
c_7_a = [ 1/4 * 1.8  + 0.1 , 1/2 * 1.8  + 0.1 , 3/4 * 1.8  + 0.1 , 1.8  + 0.1  ] 
c_8_a = [ 1/4 * 1.8  + 0.1 , 1/2 * 1.8  + 0.1 , 3/4 * 1.8  + 0.1 , 1.8  + 0.1  ] 
c_9_a = [ 1/4 * 1.8  + 0.1 , 1/2 * 1.8  + 0.1 , 3/4 * 1.8  + 0.1 , 1.8  + 0.1  ] 
c_10_a = [ 1/4 * 1.8  + 0.1 , 1/2 * 1.8  + 0.1 , 3/4 * 1.8  + 0.1 , 1.8  + 0.1  ] 
c_11_a = [ 1/4 * 1.8  + 0.1 , 1/2 * 1.8  + 0.1 , 3/4 * 1.8  + 0.1 , 1.8  + 0.1  ] 
c_12_a = [ 1/4 * 1.8  + 0.1 , 1/2 * 1.8  + 0.1 , 3/4 * 1.8  + 0.1 , 1.8  + 0.1  ] 
 

# In[7] 信道空闲概率，为随机生成，存储如下
theta_1 = 1 * 0.964
theta_2 = 1 * 0.895
theta_3 = 1 * 0.803
theta_4 = 1 * 0.756
theta_5 = 1 * 0.624
theta_6 = 1 * 0.446
theta_7 = 1 * 0.364
theta_8 = 1 * 0.683
theta_9 = 1 * 0.525
theta_10 = 1 * 0.493
theta_11 = 1 * 0.423
theta_12 = 1 * 0.234

theta_set = [ theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, theta_7, theta_8, theta_9, theta_10, theta_11, theta_12 ]


# In[9] 接入信道对应的功率等级概率。该些数据为随机生成，存储如下。该仿真设置参考 Qing Zhao 于 IEEE Transactions on Signal Processing 中 《Distributed Spectrum Sensing and Access in
# Cognitive Radio Networks With Energy Constraint》 中的仿真设置 (P785 右侧第三段)： Due to hardware and power limitations, the secondary user only has a finite number of transmission power levels. We
# assume that the user transmits at a fixed rate. Hence, to ensure successful transmission, the user has to adjust its transmission power according to the current channel fading condition...

p_1 = [ 0.13,0.11,0.37,0.39 ]
p_2 = [ 0.02,0.12,0.46,0.40 ]
p_3 = [ 0.02,0.23,0.09,0.66 ]
p_4 = [ 0.18,0.09,0.11,0.62 ]
p_5 = [ 0.04,0.28,0.24,0.44 ]
p_6 = [ 0.20,0.07,0.26,0.47 ]
p_7 = [ 0.26,0.19,0.28,0.27 ]
p_8 = [ 0.04,0.14,0.25,0.57 ]
p_9 = [ 0.27,0.39,0.31,0.03 ]
p_10 = [ 0.11,0.49,0.35,0.05 ] 
p_11 = [ 0.34,0.01,0.53,0.12 ]
p_12 = [ 0.09,0.06,0.57,0.28 ]


# In[10] 感知接入信道消耗的期望能量

c_1_ = c_1_s + np.dot(c_1_a, p_1) * theta_1 
c_2_ = c_2_s + np.dot(c_2_a, p_2) * theta_2
c_3_ = c_3_s + np.dot(c_3_a, p_3) * theta_3
c_4_ = c_4_s + np.dot(c_4_a, p_4) * theta_4
c_5_ = c_5_s + np.dot(c_5_a, p_5) * theta_5
c_6_ = c_6_s + np.dot(c_6_a, p_6) * theta_6
c_7_ = c_7_s + np.dot(c_7_a, p_7) * theta_7
c_8_ = c_8_s + np.dot(c_8_a, p_8) * theta_8
c_9_ = c_9_s + np.dot(c_9_a, p_9) * theta_9
c_10_ = c_10_s + np.dot(c_10_a, p_10) * theta_10
c_11_ = c_11_s + np.dot(c_11_a, p_11) * theta_11
c_12_ = c_12_s + np.dot(c_12_a, p_12) * theta_12

cost_set = [ c_1_, c_2_, c_3_, c_4_, c_5_, c_6_, c_7_, c_8_, c_9_, c_10_, c_11_, c_12_ ]


# In[11] 每一条信道建立一个class并存储相关信息

n_1 = channel_info(p_1, c_1_, theta_1, c_1_s, c_1_a)
n_2 = channel_info(p_2, c_2_, theta_2, c_2_s, c_2_a)
n_3 = channel_info(p_3, c_3_, theta_3, c_3_s, c_3_a)
n_4 = channel_info(p_4, c_4_, theta_4, c_4_s, c_4_a)
n_5 = channel_info(p_5, c_5_, theta_5, c_5_s, c_5_a)
n_6 = channel_info(p_6, c_6_, theta_6, c_6_s, c_6_a)
n_7 = channel_info(p_7, c_7_, theta_7, c_7_s, c_7_a)
n_8 = channel_info(p_8, c_8_, theta_8, c_8_s, c_8_a)
n_9 = channel_info(p_9, c_9_, theta_9, c_9_s, c_9_a)
n_10 = channel_info(p_10, c_10_, theta_10, c_10_s, c_10_a)
n_11 = channel_info(p_11, c_11_, theta_11, c_11_s, c_11_a)
n_12 = channel_info(p_12, c_12_, theta_12, c_12_s, c_12_a)

channel_set = [ n_1, n_2, n_3, n_4, n_5, n_6, n_7, n_8, n_9, n_10, n_11, n_12 ]


# 10 条信道的情况 
# channel_set = [ n_1, n_2, n_3, n_4, n_5, n_6, n_7, n_8, n_9, n_10 ]

N = len(channel_set) # 信道数量

 
# In[13] 求得最优信道 以及最优方法对应的最大累积吞吐量
reward_optimal = np.zeros(N)
reward_cost_ratio = np.zeros(N)
reward_optimal_value_benchmark = np.zeros(len(budget_list))
reward_sum_benchmark = 0
cost_sum_benchmark = 0

for i in range(N):
    reward_optimal[i] = channel_set[i].theta # 空闲概率
    reward_cost_ratio[i] = reward_optimal[i]/channel_set[i].cost
    
# temp_1 = sorted(reward_optimal, reverse = True)[0:M]
temp_2_value = sorted(reward_cost_ratio, reverse = True)[0:M]

mixer = np.random.random(reward_cost_ratio.size)
indices = list(np.lexsort((mixer, reward_cost_ratio))) 
output = indices[::-1]     # 降序排序
chosen_arm_ = output[0:M]  # M channels are chosen to be sensed 

for i, arm in enumerate(chosen_arm_):
    reward_sum_benchmark += reward_optimal[arm]
    cost_sum_benchmark   += channel_set[arm].cost
    
# reward_optimal_value = reward_sum_benchmark * Transmission_Rate * Transmission_Duration
for index, budget in enumerate(budget_list):
    reward_optimal_value_benchmark[index] = ( budget - varepsilon_threshold) * Transmission_Rate * Transmission_Duration * reward_sum_benchmark/cost_sum_benchmark  # 得到的 throughput benchmark 乘 transmission rate 和 duration

average_reward_set_benchmark = np.zeros(len(budget_list))  # 记录最优能量效率，即总吞吐量/budget
average_reward_set_benchmark = [ reward_optimal_value_benchmark[index]/(budget_list[index]*1e-03) for index in range(len(reward_optimal_value_benchmark)) ]
average_reward_benchmark = average_reward_set_benchmark[len(reward_optimal_value_benchmark)-1]

print("The optimal channels are", chosen_arm_)
print("The reward with budget is", reward_optimal_value_benchmark)

    

# In[14] 所提算法仿真

simulation_Num = 10000 # 该设置是为了设置记录数据的空间 存储空间长度大于停止时间 

# 每条信道相关信息更新过程 创建一个 policy 的class 
policy = []
for i in range(N):
    policy.append(algo(channel_set[i]))


sum_reward_set = np.zeros(len(budget_list))  # 记录总吞吐量
average_reward_set = np.zeros(len(budget_list))  # 记录能量效率，即总吞吐量/budget
reward_difference_set = np.zeros(len(budget_list)) # 记录吞吐量损失
budget_list_plot = np.zeros(len(budget_list))  # budget 的画图形式，将原本 budget 的 10^{-3} J 的形式转化为 J 的量纲形式


for index_, budget in enumerate(budget_list):
    
    simulation_Num_ = math.ceil(budget*3) # 该设置是为了记录数据的空间 存储空间长度大于停止时间 
    reward_monte_set = np.zeros( (Monte_Carlo_Num, simulation_Num_) )  # 每时隙记录获得的吞吐量收益
    reward_monte_mean = np.zeros(simulation_Num_)  # 记录每时隙吞吐量收益均值
    end_time_budget = np.zeros(Monte_Carlo_Num) # 记录停止时间
    sum_reward = np.zeros(Monte_Carlo_Num)  
    reward_difference = np.zeros(Monte_Carlo_Num) # 记录吞吐量损失
 
    print("The budget is :", budget*1e-03, "J")
    
    for monte in range(Monte_Carlo_Num):
        # print("This is the", monte, "simulation")
        for i in range(len(policy)):
            policy[i].initial() # 每次仿真需初始化信道相关信息
    
        reward = np.zeros(N)
        cost_ = np.zeros(N)
        index = np.zeros(N)
        channel_state = np.zeros(N)
        chosen_arm = []
            
        reward_monte = np.zeros(simulation_Num_)
        # regret_monte = np.zeros(simulation_Num_)
        
        budget_residual = budget
    
        for num in range(simulation_Num_):
            for i in range(N):
                channel_state[i] = np.random.binomial(1,theta_set[i],1) # 当前时隙中信道可用状态
                reward[i] = copy.deepcopy(channel_state[i])             # 获得的吞吐量收益
                cost_[i] = channel_set[i].c_s + channel_state[i] * random_pick( channel_set[i].c_a, channel_set[i].p ) # 消耗的能量 = 感知决策、ACK + 信道感知 + 当前信道状态对应的能量消耗  
                
                # '''
                # 考虑均值形式用于检验结果，这样的仿真速度会加快很多
                # channel_state[i] = copy.deepcopy(theta_set[i]) # np.random.binomial(1,theta_set[i],1) # 当前时隙中信道可用状态
                # reward[i] = copy.deepcopy(channel_state[i])             # 获得的吞吐量收益
                # cost_[i] = copy.deepcopy(cost_set[i]) # 消耗的能量 = 感知决策 + 信道感知 + 当前信道状态对应的能量消耗   
                # '''
                
            # print("The instantaneous channel state is", reward)
            # print("The energy consumption is", cost_)
            
            temp_reward = 0
            temp_cost = 0

            if budget_residual < varepsilon_threshold: 
                break

            
            if num < 1:
                for i in range(N): 
                    policy[i].compute_index(reward[i], num, cost_[i]) # 获得初始信息
            else:
                for i in range(N):
                    index[i] = policy[i].compute_index_update(reward[i], num) # 计算信道标号 
                
                # 对信道标号进行降序排序
                mixer = np.random.random(index.size)
                ucb_indices = list(np.lexsort((mixer, index))) 
                output = ucb_indices[::-1]
                chosen_arm = output[0:M]   

                
                '''
                chosen_arm = [ 0,1,2 ]
                '''

                # # 添加 epsilon 模块 
                # pro = np.random.random(1)
                # epsilo = 0.03
                # if pro < epsilo:
                #     chosen_arm = ucb_indices[0:M]
                # # # 
                
                for i, element in enumerate(chosen_arm):
                    policy[element].compute_index(reward[element], num, cost_[element])  # 更新信道相关估计信息
                    temp_reward += reward[element] # 更新累积吞吐量收益
                    # temp_cost += channel_set[element].cost
                    temp_cost += cost_[element] # 更新能量损失
                    
            budget_residual = budget_residual - temp_cost # 更新剩余能量
            reward_monte[num] = temp_reward 
            # 剩余能量小于阈值时停止仿真过程，在本仿真中设置为信道感知数量。可根据场景自行设置。

        end_time_budget[monte] = num
        
        # for i in range(N):
        #     print(policy[i].chosen_time)
        # print("This is the space")
           
        reward_monte_set[monte] = reward_monte * Transmission_Rate * Transmission_Duration
        
    reward_monte_mean = sum(reward_monte_set)/Monte_Carlo_Num
    
    sum_reward_set[index_] = sum(reward_monte_mean)
    average_reward_set[index_] = copy.deepcopy(sum_reward_set[index_]/(budget*1e-03) ) 
    reward_difference_set[index_] = reward_optimal_value_benchmark[index_] - sum_reward_set[index_]  # 记录benchmark得到的累积收益 - 设计算法得到的累积收益

    # 打印信道被选择的次数
    chosen_time_set = []
    for i in range(len(policy)):
        print("信道" + str(i+1) + "被选择感知的次数为： " + str(policy[i].chosen_time) )
        chosen_time_set.append(policy[i].chosen_time)


# In[15] 画图 

budget_list_plot = [ budget_list[i] * 1e-03 for i in range(len(budget_list)) ] # 将原本 budget 的 10^{-3} J 的形式转化为 J 的量纲形式

         
plt.plot(budget_list_plot, reward_difference_set) # 吞吐量损失与能量大小关系
plt.title('the relationship between regret and energy')
plt.ylabel('regret')
plt.xlabel('energy')
plt.show()

plt.plot(budget_list_plot, average_reward_set_benchmark) # 最优平均能量效率与能量大小关系
plt.plot(budget_list_plot, average_reward_set) # 平均能量效率与能量大小关系
plt.ylabel('throughput/energy')
plt.xlabel('energy')
plt.show()


# plt.plot(budget_list_plot, sum_reward_set) # 累积吞吐量与能量大小关系

# In[16] 
end = time.time() # 记录结束时间
print("程序运行时间: " + str(end-start) + "秒")







