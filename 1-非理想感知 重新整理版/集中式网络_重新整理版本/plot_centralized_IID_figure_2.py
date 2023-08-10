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

# channel_free_prob = [ 0.0499,0.6435,0.2852 ,0.7326, 0.4541,0.6913,0.3624,0.6017, 0.1354,0.9707,0.4904 ,0.8799, 0.5580, 0.4067,0.7994 ]';
 
P_d = 0.8
P_f = 0.3
U = 3
M = 4
K = 2

Num_record = [ 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000 ]

regret_P_d_07_P_f_03_U_3_M_4_K_2 =  np.array([ 24.1507,   28.1044,   33.1932,   36.4119 ,  39.4721 ,  40.2046,   44.0068,   43.8101 , 46.3396 ,  49.0369 ])

regret_P_d_08_P_f_03_U_3_M_4_K_2 =  np.array([  20.5497,   25.2678,   24.7483,   26.6894,   31.2509 ,  33.3461 ,  34.7901 ,  33.7665 ,  36.0170 ,  36.2562 ])


regret_P_d_09_P_f_03_U_3_M_4_K_2 =  np.array([ 16.1745 ,  20.2200 ,  22.5118 ,  24.1792 ,  27.1205 ,  28.1440 ,  29.3499 ,  29.4270 ,  29.8851  , 29.9546 ])


regret_P_d_08_P_f_02_U_3_M_4_K_2 =  np.array([ 11.5576 ,  13.2793 ,  15.4515 ,  15.7652 ,  16.0007 ,  16.6239,   16.7184,   17.3800 ,  17.6934 ,  18.1841 ])

regret_P_d_08_P_f_01_U_3_M_4_K_2 =  np.array([ 5.7102  ,  6.4873 ,   6.7033 ,   6.7683  ,  7.2366 ,  7.8057 ,   7.6596 ,   7.7683 ,   7.5083 ,   7.7228 ])


# regret_P_d_08_P_f_03_M_8_K_8 = np.array([ 15.9670,   20.2184 ,  23.8261 ,  25.6968 ,  28.2555,   30.1802 ,  31.0787 ,  31.0688  , 33.1177  , 33.3961] )

# regret_P_d_08_P_f_03_M_8_K_4 = np.array([ 7.8838 ,  11.2047 ,  14.0549 ,  15.4267  , 16.8497  , 17.9284 ,  18.1497 ,  18.1859 ,  19.1634  , 19.0934 ] )

# regret_P_d_08_P_f_03_M_6_K_6 = np.array([ 36.8217 ,  47.5893 ,  53.8529 ,  56.3984 ,  57.8391 ,  60.4715  , 63.2733  , 65.5005  , 66.7206 ,  68.2485  ])

# regret_P_d_08_P_f_03_M_6_K_3 = np.array([ 17.5298 ,  19.9576 ,  22.2452 ,  25.1721 ,  25.8356,   27.3448 ,  27.8180 ,  29.2371,  30.0095 ,  31.7878 ])
 

# regret_P_d_08_P_f_03_U_2_M_6_K_4 = np.array([  29.5787 ,  40.1313 ,  44.7271 ,  49.2892 ,  51.1554 ,  52.1064 ,  51.7172  , 52.8478  , 53.4496 ,  55.4681  ])


# regret_P_d_08_P_f_03_U_2_M_6_K_2 = np.array([  6.3961 ,   7.2002,    8.0399,    8.9129 ,   9.0541 ,   9.4982 ,   9.8483  , 10.4372 ,  10.5956 ,  10.6153 ])
 
 

# regret_P_d_08_P_f_03_U_2_M_6_K_6 = np.array([ 36.8217 ,  47.5893 ,  53.8529 ,  56.3984 ,  57.8391 ,  60.4715  , 63.2733  , 65.5005  , 66.7206 ,  68.2485  ])

plt.plot(Num_record, regret_P_d_07_P_f_03_U_3_M_4_K_2, '--.', label = '$P_d$=0.7,$P_f$=0.3,U=3,M=4,K=2', LineWidth = 2)

plt.plot(Num_record, regret_P_d_08_P_f_03_U_3_M_4_K_2, '--.', label = '$P_d$=0.8,$P_f$=0.3,U=3,M=4,K=2', LineWidth = 2)

plt.plot(Num_record, regret_P_d_09_P_f_03_U_3_M_4_K_2, '--.', label = '$P_d$=0.9,$P_f$=0.3,U=3,M=4,K=2', LineWidth = 2)



plt.plot(Num_record, regret_P_d_08_P_f_02_U_3_M_4_K_2, '--.', label = '$P_d$=0.8,$P_f$=0.2,U=3,M=4,K=2', LineWidth = 2)

plt.plot(Num_record, regret_P_d_08_P_f_01_U_3_M_4_K_2, '--.', label = '$P_d$=0.8,$P_f$=0.1,U=3,M=4,K=2', LineWidth = 2)

# plt.plot(Num_record, regret_P_d_08_P_f_03_M_8_K_8, '--.', label = 'U=4,M=2,K=2', LineWidth = 2)
# plt.plot(Num_record, regret_P_d_08_P_f_03_M_8_K_4, '--.', label = 'U=4,M=2,K=1', LineWidth = 2)
# plt.plot(Num_record, regret_P_d_08_P_f_03_M_6_K_6, '--.', label = 'U=3,M=2,K=2', LineWidth = 2)
# plt.plot(Num_record, regret_P_d_08_P_f_03_M_6_K_3, '--.', label = 'U=3,M=2,K=1', LineWidth = 2)
# plt.plot(Num_record, regret_P_d_08_P_f_03_U_2_M_6_K_4, '--.', label = 'U=2,M=3,K=2', LineWidth = 2)
# plt.plot(Num_record, regret_P_d_08_P_f_03_U_2_M_6_K_2, '--.', label = 'U=2,M=3,K=1', LineWidth = 2)

# plt.plot(budget, regret_U_3, '--*', label = 'U=3', LineWidth = 2)
# plt.plot(budget, regret_U_4, '--o', label = 'U=4', LineWidth = 2)
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
plt.legend(ncol=1,framealpha = 0)
plt.xlabel('slot number',fontsize=12)
plt.ylabel('normalized regret',fontsize=12)
plt.ylim([ 0, 80 ])
plt.grid()
fig.savefig('centralized_IID_figure_2_regret.eps', dpi = 600, format = 'eps')

# In[0]
import numpy as np
import random 
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

P_d = 0.8
P_f = 0.3
U = 3
M = 4
K = 2

Num_record = [ 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000 ]

reward_P_d_07_P_f_03_U_3_M_4_K_2 = np.array([   4.6325,    4.6409 ,   4.6434 ,   4.6451   , 4.6462  ,  4.6474  ,  4.6478 ,   4.6486 ,   4.6489 ,   4.6491 ])

reward_P_d_08_P_f_03_U_3_M_4_K_2 = np.array([4.7425,    4.7483 ,   4.7507 ,   4.7522 ,   4.7530  ,  4.7537  ,  4.7541 ,   4.7545 ,   4.7549  ,  4.7551 ])
reward_P_d_09_P_f_03_U_3_M_4_K_2 = np.array([ 4.8427 ,   4.8475,    4.8498 ,   4.8512,    4.8517 ,   4.8524 ,   4.8529 ,   4.8535 ,   4.8538  ,  4.8541])

reward_P_d_08_P_f_02_U_3_M_4_K_2 = np.array([  5.1252 ,   5.1293 ,   5.1306,    5.1317 ,   5.1324 ,   5.1328  ,  5.1332 ,   5.1334 ,   5.1336 ,   5.1338 ])
reward_P_d_08_P_f_01_U_3_M_4_K_2 = np.array([  5.3949 ,   5.3969 ,   5.3978,    5.3984,    5.3986 ,   5.3987 ,   5.3989 ,   5.3991,    5.3992 ,   5.3993 ])
# reward_P_d_08_P_f_03_M_8_K_8 = np.array([ 3.0653,    3.0700 ,   3.0718 ,   3.0732 ,   3.0739  ,  3.0745 ,   3.0750 ,   3.0756 ,   3.0758 ,   3.0762 ])

# reward_P_d_08_P_f_03_M_8_K_4 = np.array([ 2.8125,    2.8142 ,   2.8149 ,   2.8157 ,   2.8161  ,  2.8165  ,  2.8169 ,   2.8172,    2.8173  ,  2.8176 ])


# reward_P_d_08_P_f_03_M_6_K_6 = np.array([ 2.6909 ,   2.7049 ,   2.7101,    2.7128 ,   2.7148 ,   2.7171 ,   2.7184,    2.7193 ,   2.7206,    2.7210 ])

# reward_P_d_08_P_f_03_M_6_K_3 = np.array([ 2.3064 ,   2.3127 ,   2.3149 ,   2.3159 ,   2.3170 ,   2.3175 ,   2.3181  ,  2.3184  ,  2.3187  ,  2.3189 ])


# reward_P_d_08_P_f_03_U_2_M_6_K_4 = np.array([ 2.5943 ,   2.6017 ,   2.6062,    2.6085 ,   2.6105 ,   2.6120,    2.6133  ,  2.6141,    2.6148,    2.6152 ])

# reward_P_d_08_P_f_03_U_2_M_6_K_2 = np.array([1.7391 ,   1.7414,    1.7422  ,  1.7426,    1.7430 ,   1.7432  ,  1.7434  ,  1.7435 ,   1.7436  ,  1.7437 ])



# reward_P_d_08_P_f_03_U_2_M_6_K_6 = np.array([ 2.6961,    2.7064 ,   2.7115,    2.7151 ,   2.7175,    2.7189 ,   2.7199  ,  2.7208  ,  2.7215 ,   2.7221 ])

# sum_reward_set_M_2 = np.array([  406.19 ,  1231.815,  3309.343,  4144.023,  8313.543, 12487.394,       16662.144, 20837.545, 25015.265, 29191.866])

# sum_reward_set_M_3 = np.array([  401.277     ,  1212.01      ,  3248.323     ,  4063.961     ,        8148.799     , 12238.743     , 16326.748     , 20419.903     ,       24508.572     , 28600.87900001])

# sum_reward_set_N_8_M_1 = np.array([  372.815     ,  1141.139     ,  3081.53      ,  3859.157     ,        7753.776     , 11651.871     , 15550.915     , 19451.137     ,       23351.20400001, 27252.22700001])

# sum_reward_set_N_8_M_2 = np.array([  363.622,  1101.388,  2954.981,  3696.67 ,  7408.739, 11123.397,
#                                    14838.299, 18554.683, 22272.294, 25990.173])
# sum_reward_set_N_8_M_3 = np.array([  363.622,  1101.388,  2954.981,  3696.67 ,  7408.739, 11123.397,       14838.299, 18554.683, 22272.294, 25990.173])

# sum_reward_set_N_6_M_1 = np.array([  414.959,  1265.607,  3406.171,  4263.86 ,  8559.036, 12856.465,       17155.422, 21454.871, 25754.918, 30054.481])
# sum_reward_set_N_6_M_2 = np.array([  383.698,  1174.007,  3226.343,  4063.799,  8241.519, 12421.151,       16600.783, 20780.415, 24958.135, 29137.767])
# sum_reward_set_N_6_M_3 = np.array([  378.095,  1130.077,  3037.241,  3806.113,  7695.005, 11621.937,       15592.225, 19585.048, 23599.078, 27631.328])

optimal_reward_P_d_07_P_f_03_U_3_M_4_K_2 = np.ones( len(Num_record)) *  4.6548
optimal_reward_P_d_08_P_f_03_U_3_M_4_K_2 = np.ones( len(Num_record)) * 4.7584
optimal_reward_P_d_09_P_f_03_U_3_M_4_K_2 = np.ones( len(Num_record)) *  4.8576

optimal_reward_P_d_08_P_f_02_U_3_M_4_K_2 = np.ones( len(Num_record)) * 5.1359
optimal_reward_P_d_08_P_f_01_U_3_M_4_K_2 = np.ones( len(Num_record)) * 5.4002
# optimal_reward_P_d_08_P_f_03_M_8_K_8 = np.ones( len(Num_record)) * 3.0800
# optimal_reward_P_d_08_P_f_03_M_8_K_4 = np.ones( len(Num_record)) * 2.8198
# optimal_reward_P_d_08_P_f_03_M_6_K_6 = np.ones( len(Num_record)) * 2.73
# optimal_reward_P_d_08_P_f_03_M_6_K_3 = np.ones( len(Num_record)) *  2.3225
# optimal_reward_P_d_08_P_f_03_U_2_M_6_K_4 = np.ones( len(Num_record)) * 2.6216
# optimal_reward_P_d_08_P_f_03_U_2_M_6_K_2 = np.ones( len(Num_record)) *  1.745

plt.plot( Num_record, optimal_reward_P_d_07_P_f_03_U_3_M_4_K_2, '-o', label = '$P_d$=0.7,$P_f$=0.3,optimal', LineWidth = 2)
plt.plot( Num_record, reward_P_d_07_P_f_03_U_3_M_4_K_2, '-*', label = '$P_d$=0.7,$P_f$=0.3', LineWidth = 2 )

plt.plot( Num_record, optimal_reward_P_d_08_P_f_03_U_3_M_4_K_2, '-o', label = '$P_d$=0.8,$P_f$=0.3,optimal', LineWidth = 2)
plt.plot( Num_record, reward_P_d_08_P_f_03_U_3_M_4_K_2, '-*', label = '$P_d$=0.8,$P_f$=0.3', LineWidth = 2 )

plt.plot( Num_record, optimal_reward_P_d_09_P_f_03_U_3_M_4_K_2, '-o', label = '$P_d$=0.9,$P_f$=0.3,optimal', LineWidth = 2)
plt.plot( Num_record, reward_P_d_09_P_f_03_U_3_M_4_K_2, '-*', label = '$P_d$=0.9,$P_f$=0.3', LineWidth = 2 )




plt.plot( Num_record, optimal_reward_P_d_08_P_f_02_U_3_M_4_K_2, '-o', label = '$P_d$=0.8,$P_f$=0.2,optimal', LineWidth = 2)
plt.plot( Num_record, reward_P_d_08_P_f_02_U_3_M_4_K_2, '-*', label = '$P_d$=0.8,$P_f$=0.2', LineWidth = 2 )

plt.plot( Num_record, optimal_reward_P_d_08_P_f_01_U_3_M_4_K_2, '-o', label = '$P_d$=0.8,$P_f$=0.1,optimal', LineWidth = 2)
plt.plot( Num_record, reward_P_d_08_P_f_01_U_3_M_4_K_2, '-*', label = '$P_d$=0.8,$P_f$=0.1', LineWidth = 2 )

# plt.plot( Num_record, optimal_reward_P_d_08_P_f_03_M_6_K_3, '-o', label = 'U=3,M=2,K=1,optimal', LineWidth = 2)
# plt.plot( Num_record, reward_P_d_08_P_f_03_M_6_K_3, '-*', label = 'U=3,M=2,K=1', LineWidth = 2 )

# plt.plot( Num_record, optimal_reward_P_d_08_P_f_03_U_2_M_6_K_4, '-o', label = 'U=2,M=3,K=2,optimal', LineWidth = 2)
# plt.plot( Num_record, reward_P_d_08_P_f_03_U_2_M_6_K_4, '-*', label = 'U=2,M=3,K=2', LineWidth = 2 )

# plt.plot( Num_record, optimal_reward_P_d_08_P_f_03_U_2_M_6_K_2, '-o', label = 'U=2,M=3,K=1,optimal', LineWidth = 2)
# plt.plot( Num_record, reward_P_d_08_P_f_03_U_2_M_6_K_2, '-*', label = 'U=2,M=3,K=1', LineWidth = 2 )

# plt.plot(budget[0:4], sum_reward_set_M_2[0:4], 'b--*', label = 'N=10,M=2')
# plt.plot(budget[0:4], sum_reward_set_M_3[0:4], 'b--o', label = 'N=10,M=3')
# plt.plot(budget[0:4], sum_reward_set_N_8_M_1[0:4], 'r--.', label = 'N=8,M=1')
# plt.plot(budget[0:4], sum_reward_set_N_8_M_2[0:4], 'r--*', label = 'N=8,M=2')
# plt.plot(budget[0:4], sum_reward_set_N_8_M_3[0:4], 'r--o', label = 'N=8,M=3')
# plt.plot(budget[0:4], optimal_reward[0:4], 'k-', label = 'maximum')
plt.xlabel('slot number',fontsize=12)
plt.ylabel('average throughput',fontsize=12)
# plt.ylim([200,4500])
plt.legend(ncol=1,  framealpha = 0)
plt.grid()
plt.xlim([2000, 190000])
fig.savefig('centralized_IID_figure_2_reward.eps', dpi = 600, format = 'eps')


# In[1]

N = 10
M = 1 
chosen_time_1000 = [285, 98, 5, 8, 12, 20, 7, 4, 5, 1]
chosen_time_3000 = [992, 187, 8, 44, 3, 19, 3, 7, 7, 55]
chosen_time_8000 = [2902, 279, 190, 59, 6, 42, 10, 16, 15, 2]
chosen_time_10000 = [3910, 257, 36, 83, 6, 2, 16, 20, 20, 3]

fig = plt.figure()
# ax1 = fig.add_subplot(221)
plt.bar(['1','2','3','4','5','6','7','8','9','10'], chosen_time_1000, label = '1000')  
fig.savefig('algo_2_1J.eps', dpi = 600, format = 'eps')


fig = plt.figure()
plt.bar(['1','2','3','4','5','6','7','8','9','10'], chosen_time_3000, label = '3000') 
fig.savefig('algo_2_3J.eps', dpi = 600, format = 'eps')

fig = plt.figure()
plt.bar(['1','2','3','4','5','6','7','8','9','10'], chosen_time_8000, label = '8000')    
fig.savefig('algo_2_8J.eps', dpi = 600, format = 'eps')

fig = plt.figure()
plt.bar(['1','2','3','4','5','6','7','8','9','10'], chosen_time_10000, label = '10000')
fig.savefig('algo_2_10J.eps', dpi = 600, format = 'eps')


plt.show()

# plt.bar(['1','2','3','4','5','6','7','8','9','10'], chosen_time_100, label = '1000')  

# plt.bar(['1','2','3','4','5','6','7','8','9','10'], chosen_time_500, label = '3000')  

# plt.bar(['1','2','3','4','5','6','7','8','9','10'], chosen_time_1000, label = '8000')  

# plt.bar(['1','2','3','4','5','6','7','8','9','10'], chosen_time_3000, label = '10000')  




# fig.savefig('multiple_traffic_type.eps', dpi = 600, format = 'eps')


