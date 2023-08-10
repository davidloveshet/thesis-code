# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 20:51:34 2021

@author: xxx
"""

# In[11]
import numpy as np
import random 
import matplotlib
import matplotlib.pyplot as plt

fig, ax = plt.subplots()


import scipy.io
# data = scipy.io.loadmat('N_5_U_2_M_2_K_1_regret.mat') # 读取mat文件
Sample_Num = 100000
x_Num = list(range(Sample_Num))


N_15_regret = scipy.io.loadmat('N_15_regret.mat')['Regret_as_time'][0]
N_14_regret = scipy.io.loadmat('N_14_regret.mat')['Regret_as_time'][0]
N_13_regret = scipy.io.loadmat('N_13_regret.mat')['Regret_as_time'][0]
# P_d_08_P_f_03_regret = scipy.io.loadmat('P_d_08_P_f_03_regret.mat')['Regret_as_time'][0]
# P_d_08_P_f_01_regret = scipy.io.loadmat('P_d_08_P_f_01_regret.mat')['Regret_as_time'][0]
# P_d_08_P_f_02_regret = scipy.io.loadmat('P_d_08_P_f_02_regret.mat')['Regret_as_time'][0]
# P_d_09_P_f_02_regret = scipy.io.loadmat('P_d_09_P_f_02_regret.mat')['Regret_as_time'][0]
# U_2_M_3_K_2_regret = scipy.io.loadmat('U_2_M_3_K_2_regret.mat')['Regret_as_time'][0]
# U_3_M_2_K_1_regret = scipy.io.loadmat('U_3_M_2_K_1_regret.mat')['Regret_as_time'][0]
# U_3_M_2_K_2_regret = scipy.io.loadmat('U_3_M_2_K_2_regret.mat')['Regret_as_time'][0]
# U_4_M_2_K_1_regret = scipy.io.loadmat('U_4_M_2_K_1_regret.mat')['Regret_as_time'][0]
# U_4_M_2_K_2_regret = scipy.io.loadmat('U_4_M_2_K_2_regret.mat')['Regret_as_time'][0]
# U_4_M_1_K_1_regret = scipy.io.loadmat('U_4_M_1_K_1_regret.mat')['Regret_as_time'][0]




# N_5_U_2_M_2_K_1_regret = scipy.io.loadmat('N_5_U_2_M_2_K_1_regret.mat')['save_result2'][0]
# N_6_U_2_M_2_K_1_regret = scipy.io.loadmat('N_6_U_2_M_2_K_1_regret.mat')['save_result2'][0]
# N_6_U_2_M_2_K_2_regret = scipy.io.loadmat('N_6_U_2_M_2_K_2_regret.mat')['save_result2'][0]
# N_7_U_2_M_2_K_1_regret = scipy.io.loadmat('N_7_U_2_M_2_K_1_regret.mat')['save_result2'][0]
# N_7_U_2_M_2_K_2_regret = scipy.io.loadmat('N_7_U_2_M_2_K_2_regret.mat')['save_result2'][0]
# N_8_U_3_M_2_K_1_regret = scipy.io.loadmat('N_8_U_3_M_2_K_1_regret.mat')['save_result2'][0]
# N_8_U_3_M_2_K_2_regret = scipy.io.loadmat('N_8_U_3_M_2_K_2_regret.mat')['save_result2'][0]
# N_20_U_3_M_4_K_1_regret = scipy.io.loadmat('N_20_U_3_M_4_K_1_regret.mat')['save_result2'][0]
# N_20_U_3_M_4_K_2_regret = scipy.io.loadmat('N_20_U_3_M_4_K_2_regret.mat')['save_result2'][0]
# N_20_U_3_M_4_K_3_regret = scipy.io.loadmat('N_20_U_3_M_4_K_3_regret.mat')['save_result2'][0]
# N_20_U_3_M_4_K_4_regret = scipy.io.loadmat('N_20_U_3_M_4_K_4_regret.mat')['save_result2'][0]



plt.plot(x_Num,N_15_regret,'-', label = 'N=15', LineWidth = 2)
plt.plot(x_Num,N_14_regret,'-', label = 'N=14', LineWidth = 2)
plt.plot(x_Num,N_13_regret,'-', label = 'N=13', LineWidth = 2)
# plt.plot(x_Num,P_d_08_P_f_03_regret,'-', label = '$P_d=0.8$,$P_f=0.3$', LineWidth = 2)
# plt.plot(x_Num,P_d_08_P_f_01_regret,'-', label = '$P_d=0.8$,$P_f=0.1$', LineWidth = 2)
# plt.plot(x_Num,P_d_08_P_f_02_regret,'-', label = '$P_d=0.8$,$P_f=0.2$', LineWidth = 2)
# plt.plot(x_Num,P_d_09_P_f_02_regret,'-', label = '$P_d=0.9$,$P_f=0.2$', LineWidth = 2)
# plt.plot(x_Num,U_2_M_3_K_2_regret,'-', label = 'U=2,M=3,K=2', LineWidth = 2)
# plt.plot(x_Num,U_3_M_2_K_1_regret,'-', label = 'U=2,M=3,K=1', LineWidth = 2)
# plt.plot(x_Num,U_3_M_2_K_2_regret,'-', label = 'U=2,M=3,K=2', LineWidth = 2)
# plt.plot(x_Num,U_4_M_2_K_1_regret,'-', label = 'U=4,M=2,K=1', LineWidth = 2)
# plt.plot(x_Num,U_4_M_2_K_2_regret,'-', label = 'U=4,M=2,K=2', LineWidth = 2)
# plt.plot(x_Num,U_4_M_1_K_1_regret,'-', label = 'U=4,M=1,K=1', LineWidth = 2)
# plt.plot(x_Num,N_6_U_2_M_2_K_2_regret,'-', label = 'N=6,U=2,M=2,K=2', LineWidth = 2)
# plt.plot(x_Num,N_7_U_2_M_2_K_1_regret,'-', label = 'N=7,U=2,M=2,K=1', LineWidth = 2)
# plt.plot(x_Num,N_7_U_2_M_2_K_2_regret,'-', label = 'N=7,U=2,M=2,K=2', LineWidth = 2)
# plt.plot(x_Num,N_8_U_3_M_2_K_1_regret,'-', label = 'N=8,U=3,M=2,K=1', LineWidth = 2)
# plt.plot(x_Num,N_8_U_3_M_2_K_2_regret,'-', label = 'N=8,U=3,M=2,K=2', LineWidth = 2)
# plt.plot(x_Num,N_20_U_3_M_4_K_1_regret,'--', label = 'N=20,U=3,M=4,K=1', LineWidth = 2)
# plt.plot(x_Num,N_20_U_3_M_4_K_2_regret,'--', label = 'N=20,U=3,M=4,K=2', LineWidth = 2)
# plt.plot(x_Num,N_20_U_3_M_4_K_3_regret,'--', label = 'N=20,U=3,M=4,K=3', LineWidth = 2)
# plt.plot(x_Num,N_20_U_3_M_4_K_4_regret,'--', label = 'N=20,U=3,M=4,K=4', LineWidth = 2)


plt.xlabel('slot number',fontsize=12)
plt.ylabel('normalized regret',fontsize=12)
# plt.ylim([ -3,40])
# plt.xlim([-500, 35000])
plt.legend(ncol=2,  framealpha = 0)
plt.grid()

# fig.savefig('centralized_IID_figure_3_regret.eps', dpi = 600, format = 'eps')



# In[00]
import numpy as np
import random 
import matplotlib
import matplotlib.pyplot as plt

fig, ax = plt.subplots()


import scipy.io
# data = scipy.io.loadmat('N_5_U_2_M_2_K_1_regret.mat') # 读取mat文件
Sample_Num = 100000
x_Num = list(range(Sample_Num))


N_15_reward = scipy.io.loadmat('N_15_reward.mat')['Reward_average_as_time'][0] 
N_14_reward = scipy.io.loadmat('N_14_reward.mat')['Reward_average_as_time'][0] 
N_13_reward = scipy.io.loadmat('N_13_reward.mat')['Reward_average_as_time'][0]
# P_d_08_P_f_03_reward = scipy.io.loadmat('P_d_08_P_f_03_reward.mat')['Reward_average_as_time'][0]
# P_d_08_P_f_01_reward = scipy.io.loadmat('P_d_08_P_f_01_reward.mat')['Reward_average_as_time'][0]
# P_d_08_P_f_02_reward = scipy.io.loadmat('P_d_08_P_f_02_reward.mat')['Reward_average_as_time'][0]
# U_2_M_3_K_2_reward = scipy.io.loadmat('U_2_M_3_K_2_reward.mat')['Reward_average_as_time'][0]
# U_3_M_2_K_1_reward = scipy.io.loadmat('U_3_M_2_K_1_reward.mat')['Reward_average_as_time'][0]
# U_3_M_2_K_2_reward = scipy.io.loadmat('U_3_M_2_K_2_reward.mat')['Reward_average_as_time'][0]
# U_4_M_2_K_1_reward = scipy.io.loadmat('U_4_M_2_K_1_reward.mat')['Reward_average_as_time'][0]
# U_4_M_2_K_2_reward = scipy.io.loadmat('U_4_M_2_K_2_reward.mat')['Reward_average_as_time'][0]
# U_4_M_1_K_1_reward = scipy.io.loadmat('U_4_M_1_K_1_reward.mat')['Reward_average_as_time'][0]

N_15_reward_optimal = scipy.io.loadmat('N_15_reward_optimal.mat')['Reward_expect_all_status'][0] * np.ones(Sample_Num)
N_14_reward_optimal = scipy.io.loadmat('N_14_reward_optimal.mat')['Reward_expect_all_status'][0] * np.ones(Sample_Num)
N_13_reward_optimal = scipy.io.loadmat('N_13_reward_optimal.mat')['Reward_expect_all_status'][0] * np.ones(Sample_Num)
# P_d_08_P_f_03_reward_optimal = scipy.io.loadmat('P_d_08_P_f_03_reward_optimal.mat')['Reward_expect_all_status'][0] * np.ones(Sample_Num)
# P_d_08_P_f_01_reward_optimal = scipy.io.loadmat('P_d_08_P_f_01_reward_optimal.mat')['Reward_expect_all_status'][0] * np.ones(Sample_Num)
# P_d_08_P_f_02_reward_optimal = scipy.io.loadmat('P_d_08_P_f_02_reward_optimal.mat')['Reward_expect_all_status'][0] * np.ones(Sample_Num)
# U_2_M_3_K_2_reward_optimal = scipy.io.loadmat('U_2_M_3_K_2_reward_optimal.mat')['Reward_expect_all_status'][0] * np.ones(Sample_Num)
# U_3_M_2_K_1_reward_optimal = scipy.io.loadmat('U_3_M_2_K_1_reward_optimal.mat')['Reward_expect_all_status'][0] * np.ones(Sample_Num)
# U_3_M_2_K_2_reward_optimal = scipy.io.loadmat('U_3_M_2_K_2_reward_optimal.mat')['Reward_expect_all_status'][0] * np.ones(Sample_Num)
# U_4_M_2_K_1_reward_optimal = scipy.io.loadmat('U_4_M_2_K_1_reward_optimal.mat')['Reward_expect_all_status'][0] * np.ones(Sample_Num)
# U_4_M_2_K_2_reward_optimal = scipy.io.loadmat('U_4_M_2_K_2_reward_optimal.mat')['Reward_expect_all_status'][0] * np.ones(Sample_Num)
# U_4_M_1_K_1_reward_optimal = scipy.io.loadmat('U_4_M_1_K_1_reward_optimal.mat')['Reward_expect_all_status'][0] * np.ones(Sample_Num)


# N_5_U_2_M_2_K_1_reward = scipy.io.loadmat('N_5_U_2_M_2_K_1_reward.mat')['save_result1'][0]
# N_6_U_2_M_2_K_1_reward = scipy.io.loadmat('N_6_U_2_M_2_K_1_reward.mat')['save_result1'][0]
# N_6_U_2_M_2_K_2_reward = scipy.io.loadmat('N_6_U_2_M_2_K_2_reward.mat')['save_result1'][0]
# N_7_U_2_M_2_K_1_reward = scipy.io.loadmat('N_7_U_2_M_2_K_1_reward.mat')['save_result1'][0]
# N_7_U_2_M_2_K_2_reward = scipy.io.loadmat('N_7_U_2_M_2_K_2_reward.mat')['save_result1'][0]
# N_8_U_3_M_2_K_1_reward = scipy.io.loadmat('N_8_U_3_M_2_K_1_reward.mat')['save_result1'][0]
# N_8_U_3_M_2_K_2_reward = scipy.io.loadmat('N_8_U_3_M_2_K_2_reward.mat')['save_result1'][0]
# N_20_U_3_M_4_K_1_reward = scipy.io.loadmat('N_20_U_3_M_4_K_1_reward.mat')['save_result1'][0]
# N_20_U_3_M_4_K_2_reward = scipy.io.loadmat('N_20_U_3_M_4_K_2_reward.mat')['save_result1'][0]
# N_20_U_3_M_4_K_3_reward = scipy.io.loadmat('N_20_U_3_M_4_K_3_reward.mat')['save_result1'][0]
# N_20_U_3_M_4_K_4_reward = scipy.io.loadmat('N_20_U_3_M_4_K_4_reward.mat')['save_result1'][0]


# N_5_U_2_M_2_K_1_reward_optimal = scipy.io.loadmat('N_5_U_2_M_2_K_1_reward_optimal.mat')['Reward_expect_all'][0] * np.ones(Sample_Num)
# N_6_U_2_M_2_K_1_reward_optimal = scipy.io.loadmat('N_6_U_2_M_2_K_1_reward_optimal.mat')['Reward_expect_all'][0] * np.ones(Sample_Num)
# N_6_U_2_M_2_K_2_reward_optimal = scipy.io.loadmat('N_6_U_2_M_2_K_2_reward_optimal.mat')['Reward_expect_all'][0] * np.ones(Sample_Num)
# N_7_U_2_M_2_K_1_reward_optimal = scipy.io.loadmat('N_7_U_2_M_2_K_1_reward_optimal.mat')['Reward_expect_all'][0] * np.ones(Sample_Num)
# N_7_U_2_M_2_K_2_reward_optimal = scipy.io.loadmat('N_7_U_2_M_2_K_2_reward_optimal.mat')['Reward_expect_all'][0] * np.ones(Sample_Num)
# N_8_U_3_M_2_K_1_reward_optimal = scipy.io.loadmat('N_8_U_3_M_2_K_1_reward_optimal.mat')['Reward_expect_all'][0] * np.ones(Sample_Num) 
# N_8_U_3_M_2_K_2_reward_optimal = scipy.io.loadmat('N_8_U_3_M_2_K_2_reward_optimal.mat')['Reward_expect_all'][0] * np.ones(Sample_Num)  
# N_20_U_3_M_4_K_1_reward_optimal = scipy.io.loadmat('N_20_U_3_M_4_K_1_reward_optimal.mat')['Reward_expect_all'][0] * np.ones(Sample_Num)  
# N_20_U_3_M_4_K_2_reward_optimal = scipy.io.loadmat('N_20_U_3_M_4_K_2_reward_optimal.mat')['Reward_expect_all'][0] * np.ones(Sample_Num)  
# N_20_U_3_M_4_K_3_reward_optimal = scipy.io.loadmat('N_20_U_3_M_4_K_3_reward_optimal.mat')['Reward_expect_all'][0] * np.ones(Sample_Num)  
# N_20_U_3_M_4_K_4_reward_optimal = scipy.io.loadmat('N_20_U_3_M_4_K_4_reward_optimal.mat')['Reward_expect_all'][0] * np.ones(Sample_Num)  


plt.plot(x_Num,N_15_reward,'-', label = 'N=15', LineWidth = 2)
plt.plot(x_Num,N_14_reward,'-', label = 'N=14', LineWidth = 2)
plt.plot(x_Num,N_13_reward,'-', label = 'N=13', LineWidth = 2)
# plt.plot(x_Num,P_d_08_P_f_03_reward,'-.', label = '$P_d=0.8$,$P_f=0.3$', LineWidth = 2)
# plt.plot(x_Num,P_d_08_P_f_01_reward,'-.', label = '$P_d=0.8$,$P_f=0.1$', LineWidth = 2)
# plt.plot(x_Num,P_d_08_P_f_02_reward,'-.', label = '$P_d=0.8$,$P_f=0.2$', LineWidth = 2)
# plt.plot(x_Num,U_2_M_3_K_2_reward,'-.', label = 'U=2,M=3,K=2', LineWidth = 2)
# plt.plot(x_Num,U_3_M_2_K_1_reward,'-.', label = 'U=3,M=2,K=1', LineWidth = 2)
# plt.plot(x_Num,U_3_M_2_K_2_reward,'-.', label = 'U=3,M=2,K=2', LineWidth = 2)
# plt.plot(x_Num,U_4_M_2_K_1_reward,'-.', label = 'U=4,M=2,K=1', LineWidth = 2)
# plt.plot(x_Num,U_4_M_2_K_2_reward,'-.', label = 'U=4,M=2,K=2', LineWidth = 2)
# plt.plot(x_Num,U_4_M_1_K_1_reward,'-.', label = 'U=4,M=1,K=1', LineWidth = 2)
# plt.plot(x_Num,N_7_U_2_M_2_K_1_reward,'-.', label = 'N=7,U=2,M=2,K=1', LineWidth = 2)
# plt.plot(x_Num,N_7_U_2_M_2_K_2_reward,'-.', label = 'N=7,U=2,M=2,K=2', LineWidth = 2)
# plt.plot(x_Num,N_8_U_3_M_2_K_1_reward,'-.', label = 'N=8,U=3,M=2,K=1', LineWidth = 2)
# plt.plot(x_Num,N_8_U_3_M_2_K_2_reward,'-.', label = 'N=8,U=3,M=2,K=2', LineWidth = 2)
# plt.plot(x_Num,N_20_U_3_M_4_K_1_reward,'--', label = '(20,3,4,1)', LineWidth = 2)
# plt.plot(x_Num,N_20_U_3_M_4_K_2_reward,'--', label = '(20,3,4,2)', LineWidth = 2)
# plt.plot(x_Num,N_20_U_3_M_4_K_3_reward,'--', label = '(20,3,4,3)', LineWidth = 2)
# plt.plot(x_Num,N_20_U_3_M_4_K_4_reward,'--', label = '(20,3,4,4)', LineWidth = 2)




# plt.plot(x_Num,N_5_U_2_M_2_K_1_reward_optimal,'r-', label = '(5,2,2,1),optimal')
# plt.plot(x_Num,N_6_U_2_M_2_K_1_reward_optimal,'g-', label = '(6,2,2,1),optimal')
# plt.plot(x_Num,N_6_U_2_M_2_K_2_reward_optimal,'c-', label = '(6,2,2,2),optimal')
# plt.plot(x_Num,N_7_U_2_M_2_K_1_reward_optimal,'m-', label = '(7,2,2,1),optimal')
# plt.plot(x_Num,N_7_U_2_M_2_K_2_reward_optimal,'y-', label = '(7,2,2,2),optimal')
# plt.plot(x_Num,N_8_U_3_M_2_K_1_reward_optimal,'b-', label = '(8,3,2,1),optimal')
# plt.plot(x_Num,N_8_U_3_M_2_K_2_reward_optimal,'k-', label = '(8,3,2,2),optimal')

plt.plot(x_Num,N_15_reward_optimal,'--', label = 'N=15,optimal')
plt.plot(x_Num,N_14_reward_optimal,'--', label = 'N=14,optimal')
plt.plot(x_Num,N_13_reward_optimal,'--', label = 'N=13,optimal')
# plt.plot(x_Num,P_d_08_P_f_03_reward_optimal,'-')
# plt.plot(x_Num,P_d_08_P_f_01_reward_optimal,'-')
# plt.plot(x_Num,P_d_08_P_f_02_reward_optimal,'-')
# plt.plot(x_Num,U_2_M_3_K_2_reward_optimal,'-')
# plt.plot(x_Num,U_3_M_2_K_1_reward_optimal,'-')
# plt.plot(x_Num,U_3_M_2_K_2_reward_optimal,'-')
# plt.plot(x_Num,U_4_M_2_K_1_reward_optimal,'-')
# plt.plot(x_Num,U_4_M_2_K_2_reward_optimal,'-')
# plt.plot(x_Num,U_4_M_1_K_1_reward_optimal,'-')
# plt.plot(x_Num,N_6_U_2_M_2_K_2_reward_optimal,'-')
# plt.plot(x_Num,N_7_U_2_M_2_K_1_reward_optimal,'-')
# plt.plot(x_Num,N_7_U_2_M_2_K_2_reward_optimal,'-')
# plt.plot(x_Num,N_8_U_3_M_2_K_1_reward_optimal,'-')
# plt.plot(x_Num,N_8_U_3_M_2_K_2_reward_optimal,'-')


# plt.plot(x_Num,N_20_U_3_M_4_K_1_reward_optimal,'-', label = '(20,3,4,1),optimal')
# plt.plot(x_Num,N_20_U_3_M_4_K_2_reward_optimal,'-', label = '(20,3,4,2),optimal')
# plt.plot(x_Num,N_20_U_3_M_4_K_3_reward_optimal,'-', label = '(20,3,4,3),optimal')
# plt.plot(x_Num,N_20_U_3_M_4_K_4_reward_optimal,'-', label = '(20,3,4,4),optimal')


plt.xlabel('slot number',fontsize=12)
plt.ylabel('average throughput',fontsize=12)
plt.legend(ncol=2,  framealpha = 0)
plt.ylim([3.9,4.9 ])
plt.grid()


# fig.savefig('centralized_IID_figure_3_reward.eps', dpi = 600, format = 'eps')



 

