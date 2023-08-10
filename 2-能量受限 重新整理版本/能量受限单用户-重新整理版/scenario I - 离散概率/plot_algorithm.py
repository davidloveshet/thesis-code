# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 20:51:34 2021

@author: xxx
"""

# In[11] regret 
import numpy as np
import random 
import matplotlib
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

 
budget_list = [ 1.0, 3.0, 8.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0]

regret_M_1_N_12 = np.array([ 0.67785534, 1.45504601, 2.4664687 , 2.72124538,
       3.53780275, 4.02121613, 4.36631751, 4.62447888, 4.84460626,
       5.02729364, 5.17495301, 5.31058839, 5.43841177, 5.54195315,
       5.63582852, 5.7378759 , 5.81333728, 5.88881665, 5.94964403])


regret_M_2_N_12 = np.array([ 0.53983072, 1.09742016, 1.95920576, 2.2320352 ,
       3.1768544 , 3.76840561, 4.22214881, 4.57871001, 4.86564721,
       5.12014842, 5.33270962, 5.51855882, 5.66041602, 5.81517922,
       5.94863043, 6.08202763, 6.17850683, 6.29116803, 6.37623523])

regret_M_3_N_12 = np.array([  0.34102144, 0.64943832, 1.10287953, 1.27786241,
       1.86700882, 2.25210722, 2.61558763, 2.87992404, 3.08435645,
       3.26981686, 3.44177727, 3.56826967, 3.72550608, 3.84946049,
       3.9610309 , 4.04875131, 4.13017171, 4.23245412, 4.28509253])

regret_M_4_N_12 = np.array([  0.22960583, 0.39804548, 0.6958626 , 0.76179225,
       1.00248451, 1.20865276, 1.33074301, 1.49767126, 1.55821952,
       1.62603977, 1.70887202, 1.76202228, 1.82021253, 1.85797278,
       1.85782503, 1.89050929, 1.88449354, 1.91049979, 1.91254805])
 
plt.plot(budget_list, regret_M_1_N_12, '-^', label = 'N=12,M=1', LineWidth = '2')
plt.plot(budget_list, regret_M_2_N_12, '-o', label = 'N=12,M=2', LineWidth = '2')
plt.plot(budget_list, regret_M_3_N_12, '-.', label = 'N=12,M=3', LineWidth = '2')
plt.plot(budget_list, regret_M_4_N_12, '-*', label = 'N=12,M=4', LineWidth = '2')



# plt.plot(budget_list, regret_M_2, '--*', label = 'N=12,M=2', LineWidth = '2')
# plt.plot(budget_list, regret_M_3, '--o', label = 'N=12,M=3', LineWidth = '2')
# plt.plot(budget_list, regret_N_8_M_1, '--.', label = 'N=10,M=1', LineWidth = '2')
# plt.plot(budget_list, regret_N_8_M_2, '--*', label = 'N=10,M=2', LineWidth = '2')
# plt.plot(budget_list, regret_N_8_M_3, '--o', label = 'N=10,M=3', LineWidth = '2')
# plt.plot(budget, regret_N_6_M_1, 'b--o', label = 'N=6,M=1')
# plt.plot(budget, regret_N_6_M_2, 'b--*', label = 'N=6,M=2')
# plt.plot(budget, regret_N_6_M_3, 'b--.', label = 'N=6,M=3')
plt.legend( ncol =2, fontsize=12,framealpha = 0 )
plt.xlabel('energy (J)', fontsize=14)
plt.ylabel('regret', fontsize=14)
plt.grid()

fig.savefig('sce_1_regret.eps', dpi = 600, format = 'eps')

 


# In[0] average reward
import numpy as np
import random 
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

 

budget_list = [ 1.0, 3.0, 8.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0]



average_reward_M_1_N_12 = np.array([  8.191152  , 8.383992  , 8.56069875, 8.5968828 ,
       8.6921172 , 8.7349668 , 8.7598494 , 8.77651776, 8.7882639 ,
       8.79718886, 8.80432042, 8.8100008 , 8.81462322, 8.81862595,
       8.8220421 , 8.82486983, 8.8274835 , 8.82974856, 8.83182206])

average_reward_M_1_N_12_benchmark = np.array([   8.86900733768265,  8.869007337682648,  8.86900733768265, 8.86900733768265, 8.86900733768265, 8.86900733768265, 8.86900733768265,
 8.869007337682648, 8.86900733768265, 8.86900733768265, 8.86900733768265, 8.86900733768265, 8.869007337682648, 8.869007337682648, 8.86900733768265, 8.869007337682648,
 8.86900733768265, 8.869007337682648, 8.86900733768265])

average_reward_M_2_N_12 = np.array([  8.233038  , 8.407062  , 8.527968  , 8.5496652 ,
       8.614026  , 8.6472552 , 8.667315  , 8.68129452, 8.6917746 ,
       8.69972374, 8.70620985, 8.7115514 , 8.71626456, 8.72000345,
       8.7232968 , 8.72608389, 8.72873653, 8.7309276 , 8.73301725])

average_reward_M_2_N_12_benchmark = 8.772868720215795 * np.ones(len(budget_list))


average_reward_M_3_N_12 = np.array([  8.242218  , 8.36676   , 8.4453795 , 8.4554532 ,
       8.489889  , 8.5081692 , 8.51784975, 8.52564096, 8.5318335 ,
       8.53652777, 8.54021722, 8.543592  , 8.54598438, 8.54824435,
       8.55023085, 8.5520952 , 8.55373821, 8.55502308, 8.55645761])

average_reward_M_3_N_12_benchmark = 8.58323944081731 * np.ones(len(budget_list))


average_reward_M_4_N_12 = np.array([  8.201754  , 8.298678  , 8.344377  , 8.3551806 ,
       8.3812356 , 8.3910714 , 8.39809125, 8.4014064 , 8.4053895 ,
       8.40813069, 8.40999892, 8.4117818 , 8.4131577 , 8.41446916,
       8.41587795, 8.41681745, 8.41789916, 8.41862316, 8.4194064 ])

average_reward_M_4_N_12_benchmark = 8.431359825286469 * np.ones(len(budget_list))



plt.plot(budget_list , average_reward_M_1_N_12_benchmark, '--', label = 'N=12,M=1,optimal', Linewidth = '2')
plt.plot(budget_list , average_reward_M_1_N_12, '-^', label = 'N=12,M=1', Linewidth = '2')
plt.plot(budget_list , average_reward_M_2_N_12_benchmark, '--', label = 'N=12,M=2,optimal', Linewidth = '2')
plt.plot(budget_list , average_reward_M_2_N_12, '-o', label = 'N=12,M=2', Linewidth = '2')
plt.plot(budget_list , average_reward_M_3_N_12_benchmark, '--', label = 'N=12,M=3,optimal', Linewidth = '2')
plt.plot(budget_list , average_reward_M_3_N_12, '-.', label = 'N=12,M=3', Linewidth = '2')
plt.plot(budget_list , average_reward_M_4_N_12_benchmark, '--', label = 'N=12,M=4,optimal', Linewidth = '2')
plt.plot(budget_list , average_reward_M_4_N_12, '-*', label = 'N=12,M=4', Linewidth = '2')







# sum_reward_set_M_1 = np.array([  477.181     ,  1459.788     ,  3929.767     ,  4918.724     ,        9869.08      , 14821.371     , 19774.902     , 24728.118     ,       29683.00900001, 34636.82400002])
# sum_reward_set_M_1_ = [ sum_reward_set_M_1[i]/budget_list[i] for i in range(len(budget_list)) ]
# optimal_M_12_1 = np.ones(len(budget_list)) * 495.487


# sum_reward_set_M_2 = np.array([  466.574     ,  1416.067     ,  3801.979     ,  4757.932     ,        9542.332     , 14329.183     , 19117.52      , 23905.31      ,       28694.69900001, 33485.21200001])
# sum_reward_set_M_2_ = [ sum_reward_set_M_2[i]/budget_list[i] for i in range(len(budget_list)) ]
# optimal_M_12_2 = np.ones(len(budget_list)) * 478.969375

# sum_reward_set_M_3 = np.array([  456.956     ,  1377.847     ,  3692.756     ,  4617.715     ,        9254.355     , 13892.87      , 18535.296     , 23177.294     ,       27818.52      , 32460.69100001])
# sum_reward_set_M_3_ = [ sum_reward_set_M_3[i]/budget_list[i] for i in range(len(budget_list)) ]
# optimal_M_12_3 = np.ones(len(budget_list)) * 464.548

# sum_reward_set_N_8_M_1 = np.array([  417.519,  1275.172,  3433.25 ,  4298.888,  8631.756, 12968.185,       17306.538, 21645.974, 25985.647, 30325.992])
# sum_reward_set_N_8_M_1_ = [ sum_reward_set_N_8_M_1[i]/budget_list[i] for i in range(len(budget_list)) ]
# optimal_M_10_1 = np.ones(len(budget_list)) * 434.22225




# sum_reward_set_N_8_M_2 = np.array([  417.577     ,  1266.519     ,  3402.593     ,  4258.215     ,        8538.747     , 12822.668     , 17106.935     , 21392.786     ,       25677.545     , 29963.34600001])
# sum_reward_set_N_8_M_2_ = [ sum_reward_set_N_8_M_2[i]/budget_list[i] for i in range(len(budget_list)) ]
# optimal_M_10_2 = np.ones(len(budget_list)) *428.72

# sum_reward_set_N_8_M_3 = np.array([  412.206     ,  1248.394     ,  3341.297     ,  4178.638     ,        8370.028     , 12565.648     , 16759.532     , 20956.918     ,       25152.67000001, 29350.70500001])
# sum_reward_set_N_8_M_3_ = [ sum_reward_set_N_8_M_3[i]/budget_list[i] for i in range(len(budget_list)) ]
# optimal_M_10_3 = np.ones(len(budget_list)) *419.95462

# # sum_reward_set_N_6_M_1 = np.array([  414.959,  1265.607,  3406.171,  4263.86 ,  8559.036, 12856.465,       17155.422, 21454.871, 25754.918, 30054.481])
# # sum_reward_set_N_6_M_2 = np.array([  383.698,  1174.007,  3226.343,  4063.799,  8241.519, 12421.151,       16600.783, 20780.415, 24958.135, 29137.767])
# # sum_reward_set_N_6_M_3 = np.array([  378.095,  1130.077,  3037.241,  3806.113,  7695.005, 11621.937,       15592.225, 19585.048, 23599.078, 27631.328])

# optimal_reward = np.array([  430.18153661,  1290.54460983,  3441.45229287,  4301.81536608,        8603.63073217, 12905.44609825, 17207.26146434, 21509.07683042,       25810.89219651, 30112.70756259])

# plt.plot(budget_list , average_reward_M_1_N_12_benchmark, '-', label = 'N=12,M=1,optimal', Linewidth = '2')
# plt.plot(budget_list , average_reward_M_1_N_12, '--.', label = 'N=12,M=1', Linewidth = '2')


# plt.plot(budget_list , sum_reward_set_M_2_, '--*', label = 'N=12,M=2', Linewidth = '2')
# plt.plot(budget_list , sum_reward_set_M_3_, '--o', label = 'N=12,M=3', Linewidth = '2')

# plt.plot(budget_list , optimal_M_12_2, '-', label = 'N=12,M=2,optimal', Linewidth = '2')
# plt.plot(budget_list , optimal_M_12_3, '-', label = 'N=12,M=3,optimal', Linewidth = '2')

# plt.plot(budget_list , sum_reward_set_N_8_M_1_, '--.', label = 'N=10,M=1', Linewidth = '2')
# plt.plot(budget_list , sum_reward_set_N_8_M_2_, '--*', label = 'N=10,M=2', Linewidth = '2')
# plt.plot(budget_list , sum_reward_set_N_8_M_3_, '--o', label = 'N=10,M=3', Linewidth = '2')
# plt.plot(budget_list , optimal_M_10_1, '-', label = 'N=10,M=1,optimal', Linewidth = '2')
# plt.plot(budget_list , optimal_M_10_2, '-', label = 'N=10,M=2,optimal', Linewidth = '2')
# plt.plot(budget_list , optimal_M_10_3, '-', label = 'N=10,M=3,optimal', Linewidth = '2')
 
plt.xlabel('energy (J)',fontsize=14)
plt.ylabel('throughput/energy (M bits/J)',fontsize=14)
# plt.ylim([300, 500])
plt.xlim([-5,180])
plt.legend(ncol=2, fontsize= 10.5, framealpha = 0)
plt.grid()
fig.savefig('sce_1_average_reward.eps', dpi = 600, format = 'eps')

 


