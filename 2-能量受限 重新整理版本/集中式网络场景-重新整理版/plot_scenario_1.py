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

 

budget = [ 2, 5, 10, 20,  40,  60,  80,  100 , 150, 200 , 250  ] 

regret_U_2 = np.array([ 11.48831322,  26.56981217,  35.4796183 ,  54.42937462, 69.78662576,  81.2582143 ,  91.0193344 ,  95.53671281,  98.32127587, 101.04959342, 102.79831155])

regret_U_3 = np.array([ 15.60091582,  29.45552476,  54.24702062,  76.09954494, 106.8941055 , 125.89963313, 140.56577081, 156.9839343 , 182.19987867, 196.48570463, 200.6554046 ])

regret_U_4 = np.array([ 14.38896214,  30.57230329,  63.54867518,  94.27528511, 137.94449441, 166.64770839, 193.47756146, 212.2946283 , 247.14003157, 275.32471543, 286.5062085])

regret_U_5 = np.array([ 25.22169126,  47.93571082,  77.18547037, 123.43005408, 153.9029739 , 189.67686798, 214.90592442, 239.56519205, 268.26668476, 295.9991367 , 308.82639])

regret_U_6 = np.array([ 30.64177035,  57.41043466,  96.36251305, 131.85952   , 167.07086479, 203.29071899, 234.6656576 , 253.8392669 , 285.9410753 , 308.8730174 , 318.39190455])


plt.plot(budget, regret_U_2, '--.', label = 'U=2', LineWidth = 2)
plt.plot(budget, regret_U_3, '--*', label = 'U=3', LineWidth = 2)
plt.plot(budget, regret_U_4, '--o', label = 'U=4', LineWidth = 2)
plt.plot(budget, regret_U_5, '--o', label = 'U=5', LineWidth = 2)
plt.plot(budget, regret_U_6, '--o', label = 'U=6', LineWidth = 2)
# plt.plot(budget, regret_U_5, '--^', label = 'U=5', LineWidth = 2)
# plt.plot(budget, regret_U_6, '--+', label = 'U=6', LineWidth = 2)
# plt.plot(budget, regret_M_2, 'b--*', label = 'N=10,M=2')
# plt.plot(budget, regret_M_3, 'b--o', label = 'N=10,M=3')
# plt.plot(budget, regret_N_8_M_1, 'r--.', label = 'N=8,M=1')
# plt.plot(budget, regret_N_8_M_2, 'r--*', label = 'N=8,M=2')
# plt.plot(budget, regret_N_8_M_3, 'r--o', label = 'N=8,M=3')
# plt.plot(budget, regret_N_6_M_1, 'b--o', label = 'N=6,M=1')
# plt.plot(budget, regret_N_6_M_2, 'b--*', label = 'N=6,M=2')
# plt.plot(budget, regret_N_6_M_3, 'b--.', label = 'N=6,M=3')
plt.legend(ncol=2,framealpha = 0, fontsize = 12)
plt.xlabel('energy (J)', fontsize = 14)
plt.ylabel('regret', fontsize = 14)
plt.grid()
# fig.savefig('algo_centralized_scenario_I.eps', dpi = 600, format = 'eps')

# In[0]
import numpy as np
import random 
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

N = 10
M = 1 
budget = [ 2, 5, 10, 20,  40,  60,  80,  100 , 150, 200 , 250  ] 

sum_reward_set_M_2 = np.array([20.75301179, 21.18320597, 22.94920657, 23.57569967, 24.75250276, 25.14286483, 25.35942672, 25.54180127, 25.87502656, 25.98692043, 26.08597515])
optimal_M_2 = np.ones(len(budget)) * 26.5335994
# sum_reward_set_M_2 = np.aarray([20.75301179, 21.18320597, 22.94920657, 23.57569967, 24.75250276, 25.14286483, 25.35942672, 25.54180127, 25.87502656, 25.98692043, 26.08597515])

sum_reward_set_M_3 = np.array([29.17474539, 31.08409834, 31.55050123, 33.17022605, 34.30285066, 34.94354274, 35.28063116, 35.40536395, 35.87605611, 35.96277477, 36.17258168])
optimal_M_3 = np.ones(len(budget)) * 37.08999822
# sum_reward_set_N_8_M_1 = np.array([  372.815     ,  1141.139     ,  3081.53      ,  3859.157     ,        7753.776     , 11651.871     , 15550.915     , 19451.137     ,       23351.20400001, 27252.22700001])


sum_reward_set_M_4 = np.array([38.84798311, 39.92800352, 40.68759666, 42.32869992, 43.59385182, 44.43166904, 44.62399466, 44.9195179 , 45.39486397, 45.75337964, 45.91604929])
optimal_M_4 = np.ones(len(budget)) * 47.10375832

sum_reward_set_M_5 = np.array([51.33350586, 54.35720932, 56.22580445, 57.77284878, 60.09677714, 60.78307036, 61.25802743, 61.54869957, 62.15590692, 62.4643558 , 62.60154603])
optimal_M_5 = np.ones(len(budget)) * 63.85491978

sum_reward_set_M_6 = np.array([65.61200612, 69.79281234, 71.98059946, 74.87569475, 77.15545857, 77.73106403, 78.3181246 , 78.44464048, 79.369905  , 79.8211397 , 79.97240861])

optimal_M_6 = np.ones(len(budget)) * 81.21940127

# sum_reward_set_N_8_M_2 = np.array([  363.622,  1101.388,  2954.981,  3696.67 ,  7408.739, 11123.397,
#                                    14838.299, 18554.683, 22272.294, 25990.173])
# sum_reward_set_N_8_M_3 = np.array([  363.622,  1101.388,  2954.981,  3696.67 ,  7408.739, 11123.397,       14838.299, 18554.683, 22272.294, 25990.173])

# sum_reward_set_N_6_M_1 = np.array([  414.959,  1265.607,  3406.171,  4263.86 ,  8559.036, 12856.465,       17155.422, 21454.871, 25754.918, 30054.481])
# sum_reward_set_N_6_M_2 = np.array([  383.698,  1174.007,  3226.343,  4063.799,  8241.519, 12421.151,       16600.783, 20780.415, 24958.135, 29137.767])
# sum_reward_set_N_6_M_3 = np.array([  378.095,  1130.077,  3037.241,  3806.113,  7695.005, 11621.937,       15592.225, 19585.048, 23599.078, 27631.328])

# optimal_reward = np.array([  430.18153661,  1290.54460983,  3441.45229287,  4301.81536608,        8603.63073217, 12905.44609825, 17207.26146434, 21509.07683042,       25810.89219651, 30112.70756259])

plt.plot(budget, sum_reward_set_M_2, '--', label = 'N=10,M=2', LineWidth = '2')
plt.plot(budget, sum_reward_set_M_3, '--', label = 'N=10,M=3', LineWidth = '2')
plt.plot(budget, sum_reward_set_M_4, '--', label = 'N=10,M=4', LineWidth = '2')
plt.plot(budget, sum_reward_set_M_5, '--', label = 'N=10,M=5', LineWidth = '2')
plt.plot(budget, sum_reward_set_M_6, '--', label = 'N=10,M=6', LineWidth = '2')

plt.plot(budget, optimal_M_2, '-', label = 'M=2,optimal', LineWidth = '2')
plt.plot(budget, optimal_M_3, '-', label = 'M=3,optimal', LineWidth = '2')
plt.plot(budget, optimal_M_4, '-', label = 'M=4,optimal', LineWidth = '2')
plt.plot(budget, optimal_M_5, '-', label = 'M=5,optimal', LineWidth = '2')
plt.plot(budget, optimal_M_6, '-', label = 'M=6,optimal', LineWidth = '2')

# plt.plot(budget[0:4], sum_reward_set_N_8_M_1[0:4], 'r--.', label = 'N=8,M=1')
# plt.plot(budget[0:4], sum_reward_set_N_8_M_2[0:4], 'r--*', label = 'N=8,M=2')
# plt.plot(budget[0:4], sum_reward_set_N_8_M_3[0:4], 'r--o', label = 'N=8,M=3')
# plt.plot(budget[0:4], optimal_reward[0:4], 'k-', label = 'maximum')
plt.xlabel('energy (J)',fontsize=14)
plt.ylabel('throughput/energy (M/J)',fontsize=14)
# plt.ylim([200,4500])
plt.xlim([-10,400])
plt.grid()
plt.legend(ncol=1, fontsize=11,framealpha = 0)

# fig.savefig('algo_centralized_scenario_I_average_reward.eps', dpi = 600, format = 'eps')


# In[1]

# N = 10
# M = 1 
# chosen_time_1000 = [285, 98, 5, 8, 12, 20, 7, 4, 5, 1]
# chosen_time_3000 = [992, 187, 8, 44, 3, 19, 3, 7, 7, 55]
# chosen_time_8000 = [2902, 279, 190, 59, 6, 42, 10, 16, 15, 2]
# chosen_time_10000 = [3910, 257, 36, 83, 6, 2, 16, 20, 20, 3]

# fig = plt.figure()
# # ax1 = fig.add_subplot(221)
# plt.bar(['1','2','3','4','5','6','7','8','9','10'], chosen_time_1000, label = '1000')  
# fig.savefig('algo_2_1J.eps', dpi = 600, format = 'eps')


# fig = plt.figure()
# plt.bar(['1','2','3','4','5','6','7','8','9','10'], chosen_time_3000, label = '3000') 
# fig.savefig('algo_2_3J.eps', dpi = 600, format = 'eps')

# fig = plt.figure()
# plt.bar(['1','2','3','4','5','6','7','8','9','10'], chosen_time_8000, label = '8000')    
# fig.savefig('algo_2_8J.eps', dpi = 600, format = 'eps')

# fig = plt.figure()
# plt.bar(['1','2','3','4','5','6','7','8','9','10'], chosen_time_10000, label = '10000')
# fig.savefig('algo_2_10J.eps', dpi = 600, format = 'eps')


# plt.show()

# plt.bar(['1','2','3','4','5','6','7','8','9','10'], chosen_time_100, label = '1000')  

# plt.bar(['1','2','3','4','5','6','7','8','9','10'], chosen_time_500, label = '3000')  

# plt.bar(['1','2','3','4','5','6','7','8','9','10'], chosen_time_1000, label = '8000')  

# plt.bar(['1','2','3','4','5','6','7','8','9','10'], chosen_time_3000, label = '10000')  




# fig.savefig('multiple_traffic_type.eps', dpi = 600, format = 'eps')


