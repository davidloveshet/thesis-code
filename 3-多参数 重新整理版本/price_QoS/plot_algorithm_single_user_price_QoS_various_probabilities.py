# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 20:51:34 2021

@author: xxx
"""

# In[11] s = 1,2,3 regret 
import numpy as np
import random 
import matplotlib
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
 
T_simulation = 10000
regret_x_label = [ round(0.03 * T_simulation), round(0.05 * T_simulation), round(0.1 * T_simulation), round(0.2 * T_simulation), round(0.3 *T_simulation), round(0.4 *T_simulation), round(0.5 *T_simulation), round(0.6 *T_simulation), round(0.7 *T_simulation), round(0.8 *T_simulation), round(0.9 *T_simulation),  T_simulation  ]

  
cumulative_utility_regret_y_label_proposed_scenario_I = np.array([ 65.55     ,  90.2      , 129.4      , 178.25     , 311.0518664,
       366.8296289, 411.9004864, 441.1233839, 598.2940364, 656.8242389,
       696.8086226, 723.4882127])


cumulative_utility_regret_y_label_greedy_scenario_I = np.array([ 67.       ,  95.8      , 161.2      , 238.6      , 387.3534682,
       479.9348082, 542.4843382, 592.7700782, 751.3386674, 828.8877386,
       894.0229622, 956.8162706])

 
cumulative_utility_regret_y_label_random_scenario_I = np.array([ 195.6      ,  332.       ,  655.2      , 1314.8      ,
       1792.3673258, 2273.0756958, 2736.4328258, 3199.4552258,
       3721.6118066, 4243.9833458, 4765.2178682, 5290.2254762])

cumulative_utility_regret_y_label_fixed_scenario_I = np.array([ 288.      ,  479.6     ,  963.      , 1929.      , 2495.377443,
       3070.862763, 3625.199903, 4199.909143, 4370.037403, 4538.044363,
       4703.364343, 4867.977223])


cumulative_utility_regret_y_label_proposed_scenario_II = np.array([104.9048418, 148.7012324, 232.3281786, 339.5234562, 412.3470574,
       478.494645 , 531.1595332, 586.9678306, 629.5555914, 660.398836 ,
       702.6834588, 733.3553344])

cumulative_utility_regret_y_label_random_scenario_II = np.array([ 169.3047674,  273.3865528,  549.4523866, 1086.6017012,
       1624.1930822, 2176.3810542, 2739.9662146, 3269.5159958,
       3821.4128844, 4356.9283018, 4900.492556 , 5450.970727 ])

cumulative_utility_regret_y_label_greedy_scenario_II = np.array([ 95.3722972, 139.6701634, 222.8196598, 359.2291934, 454.6433662,
       555.1361514, 645.6525032, 718.2139638, 793.518565 , 853.8253162,
       904.1902098, 961.9983368])

cumulative_utility_regret_y_label_fixed_scenario_II = np.array([ 162.004398 ,  258.9481202,  516.056309 , 1033.597428 ,
       1565.7702548, 2107.491531 , 2625.7096464, 3128.0063272,
       3642.3069434, 4169.0945562, 4710.1788568, 5227.113366 ])


# plt.plot(regret_x_label, cumulative_utility_regret_y_label_M_1_K_1,'-o', label = 'M=1,K=1' )
# plt.plot(regret_x_label, cumulative_utility_regret_y_label_M_2_K_1,'-.', label = 'M=2,K=1' )
# plt.plot(regret_x_label, cumulative_utility_regret_y_label_M_2_K_2,'-^', label = 'M=2,K=2' )
# plt.plot(regret_x_label, cumulative_utility_regret_y_label_M_3_K_1,'--o', label = 'M=3,K=1' )
# plt.plot(regret_x_label, cumulative_utility_regret_y_label_M_3_K_2,'--.', label = 'M=3,K=2' )
plt.plot(regret_x_label, cumulative_utility_regret_y_label_proposed_scenario_I,'-^', label = 'proposed-I' , Linewidth = 2 )
plt.plot(regret_x_label, cumulative_utility_regret_y_label_greedy_scenario_I,'-o', label = '$\epsilon-$greedy-I' , Linewidth = 2 )
plt.plot( [300, 500, 1000, 1500], np.array([195.6, 332. , 655.2, 981.8]),'-*', label = 'random-I' , Linewidth = 2 )
plt.plot( [300, 500, 1000, 1500], np.array([ 288. ,  479.6,  963. , 1455.4]),'-x', label = 'fixed-I' , Linewidth = 2 )

plt.plot(regret_x_label, cumulative_utility_regret_y_label_proposed_scenario_II,'--^', label = 'proposed-II', Linewidth = 2 )
plt.plot(regret_x_label, cumulative_utility_regret_y_label_greedy_scenario_II,'--o', label = '$\epsilon-$greedy-II' , Linewidth = 2 )
plt.plot( [300, 500, 1000, 1500], np.array([169.3047674, 273.3865528, 549.4523866, 817.366853]),'--*', label = 'random-II' , Linewidth = 2 )
plt.plot( [300, 500, 1000, 1500], [158.945332 , 262.0871846, 524.5917604, 804.7173896],'--x', label = 'fixed-II' , Linewidth = 2 )


plt.legend(ncol=2,framealpha = 0,fontsize=12)
plt.xlabel('time',fontsize=16)
plt.ylabel('utility loss',fontsize=16)
# plt.title('s=2',fontsize=16)
# plt.xlim([-50, 10000])
plt.grid()
fig.savefig('regret_QoS_various_prob.eps', dpi = 600, format = 'eps')

 

# In[] average throughput 
import numpy as np
import random 
import matplotlib
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
feature_generation = np.array([   [1], [0]  ])
T_simulation = 10000
regret_x_label = [ round(0.03 * T_simulation), round(0.05 * T_simulation),  round(0.1 * T_simulation), round(0.2 * T_simulation), round(0.3 *T_simulation), round(0.4 *T_simulation), round(0.5 *T_simulation), round(0.6 *T_simulation), round(0.7 *T_simulation), round(0.8 *T_simulation), round(0.9 *T_simulation),  T_simulation  ]

optimal_throughput_M_3_s_3 = 1.96 * np.ones(len(regret_x_label)) # M bits 
average_throughput_M_3_s_3 = np.array([1.830667, 1.832933, 1.858067, 1.8786  , 1.888   ,
       1.896767, 1.902533, 1.907633, 1.912057, 1.915575,
       1.917822, 1.9214  ])

optimal_throughput_M_3_s_2 = 2.25 * np.ones(len(regret_x_label)) # M bits 
average_throughput_M_3_s_2 = np.array([2.073   , 2.1084  , 2.1607  , 2.20095 , 2.2239  ,
       2.234725, 2.24282 , 2.247   , 2.251643, 2.253725,
       2.256633, 2.25664 ])

optimal_throughput_M_3_s_1 = 2.4 * np.ones(len(regret_x_label)) # M bits 
average_throughput_M_3_s_1 = np.array([ 2.17    , 2.21375 , 2.273875, 2.318125, 2.333875,
       2.345469, 2.352025, 2.357917, 2.361571, 2.365   ,
       2.369014, 2.370113])

optimal_throughput_M_2_s_1 = 1.65 * np.ones(len(regret_x_label)) # M bits 
average_throughput_M_2_s_1 = np.array([1.43822222, 1.47746667, 1.52546667, 1.56061667, 1.57645556,
       1.588175  , 1.59627333, 1.60162222, 1.60708571, 1.61095833,
       1.61358519, 1.61607667])

optimal_throughput_M_2_s_2 = 1.5 * np.ones(len(regret_x_label)) # M bits 
average_throughput_M_2_s_2 = np.array([1.36022222, 1.38386667, 1.42233333, 1.46243333, 1.48364444,
       1.50021667, 1.51048   , 1.51916667, 1.52530476, 1.53229167,
       1.53688889, 1.54102667])

optimal_throughput_M_2_s_3 = 1.21 * np.ones(len(regret_x_label)) # M bits 
average_throughput_M_2_s_3 = np.array([1.20022222, 1.21706667, 1.23333333, 1.24423333, 1.24253333,
       1.24403333, 1.24165333, 1.24061111, 1.23977143, 1.2384    ,
       1.23680741, 1.23655333])


# average_throughput_M_1_s_2 = np.array([0.00669867, 0.0068248 , 0.007062  , 0.0072944 , 0.007436  ,
#        0.0075346 , 0.00758912, 0.00764187, 0.00767589, 0.0077098 ,
#        0.00772973, 0.00775192])


# plt.plot(regret_x_label, average_throughput_M_1_K_1,'-o', label = 'M=1,K=1' )
# plt.plot(regret_x_label, average_throughput_M_2_K_1,'-.', label = 'M=2,K=1' )
# plt.plot(regret_x_label, average_throughput_M_2_K_2,'-^', label = 'M=2,K=2' )
# plt.plot(regret_x_label, average_throughput_M_3_K_1,'--o', label = 'M=3,K=1' )
# plt.plot(regret_x_label, average_throughput_M_3_K_2,'--.', label = 'M=3,K=2' )
plt.plot(regret_x_label, optimal_throughput_M_3_s_1,'-', label = 's=1,M=3,optimal' )
plt.plot(regret_x_label, average_throughput_M_3_s_1,'-^', label = 's=1,M=3' )
plt.plot(regret_x_label, optimal_throughput_M_3_s_2,'-', label = 's=2,M=3,optimal' )
plt.plot(regret_x_label, average_throughput_M_3_s_2,'-*', label = 's=2,M=3' )
plt.plot(regret_x_label, optimal_throughput_M_3_s_3,'-', label = 's=3,M=3,optimal' )
plt.plot(regret_x_label, average_throughput_M_3_s_3,'-o', label = 's=3,M=3' )

# plt.plot(regret_x_label, optimal_throughput_M_2_s_1,'-', label = 's=1,M=2,optimal' )
# plt.plot(regret_x_label, average_throughput_M_2_s_1,'-o', label = 's=1,M=2' )
# plt.plot(regret_x_label, optimal_throughput_M_2_s_2,'-', label = 's=2,M=2,optimal' )
# plt.plot(regret_x_label, average_throughput_M_2_s_2,'-o', label = 's=2,M=2' )
# plt.plot(regret_x_label, optimal_throughput_M_2_s_3,'-', label = 's=3,M=2,optimal' )
# plt.plot(regret_x_label, average_throughput_M_2_s_3,'-o', label = 's=3,M=2' )

# plt.plot(regret_x_label, average_throughput_M_2_s_2,'--^', label = 'M=2,s=2' )
# plt.plot(regret_x_label, average_throughput_M_1_s_2,'--^', label = 'M=1,s=2' )
# optimal_M_1_K_1 = 0.85 * np.ones(len(average_throughput_M_1_K_1))
# optimal_M_2_K_1 = 0.85 * np.ones(len(average_throughput_M_2_K_1))
# optimal_M_2_K_2 = 1.65 * np.ones(len(average_throughput_M_2_K_2))
# optimal_M_2_K_3 = 2.4 * np.ones(len(average_throughput_M_2_K_2))
# plt.plot(regret_x_label, optimal_M_1_K_1,'-', label = 'optimal,K=1' )
# # plt.plot(regret_x_label, optimal_M_2_K_1,'-', label = 'optimal average throughput, M=2,K=1' )
# plt.plot(regret_x_label, optimal_M_2_K_2,'-', label = 'optimal,K=2' )
# plt.plot(regret_x_label, optimal_M_2_K_3,'-', label = 'optimal,K=3' )

plt.legend(ncol=2,framealpha = 0,fontsize = 11.5)
plt.xlabel('time',fontsize = 16)
plt.ylabel('average throughput (Mbps)',fontsize = 16)
# plt.title('s=1',fontsize = 14)
# plt.ylim([0,2.6])
plt.xlim([-20, 16000])
plt.grid()
# fig.savefig('QoS_average_throughput.eps', dpi = 600, format = 'eps')

  
# In[] average cost 
import numpy as np
import random 
import matplotlib
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
 

regret_x_label = [ round(0.03 * T_simulation), round(0.05 * T_simulation),  round(0.1 * T_simulation), round(0.2 * T_simulation), round(0.3 *T_simulation), round(0.4 *T_simulation), round(0.5 *T_simulation), round(0.6 *T_simulation), round(0.7 *T_simulation), round(0.8 *T_simulation), round(0.9 *T_simulation),  T_simulation  ]

average_cost_M_3_s_1 = np.array([1.24740833, 1.3248875 , 1.41038375, 1.46563125, 1.48539958,
       1.49846469, 1.50553475, 1.51189542, 1.51605   , 1.52000031,
       1.52397319, 1.52587237])

average_cost_M_3_s_2 = np.array([1.00184333, 1.034292  , 1.072631  , 1.1063795 , 1.11060233,
       1.11658525, 1.120383  , 1.123204  , 1.130137  , 1.12582075,
       1.12097633, 1.113806  ])

 
average_cost_M_3_s_3 = np.array([0.5284    , 0.496372  , 0.46331   , 0.44552467, 0.43974267,
       0.43632867, 0.43356773, 0.43198678, 0.431044  , 0.43027742,
       0.42955037, 0.42925813])

average_cost_M_2_s_1 = np.array([0.82826889, 0.88163867, 0.95902533, 1.01647317, 1.04802878,
       1.07275075, 1.092747  , 1.10719278, 1.12034676, 1.13128233,
       1.1372057 , 1.1424857 ])

average_cost_M_2_s_2 = np.array([0.65169556, 0.671452  , 0.68972467, 0.718495  , 0.75415556,
       0.77457083, 0.79035707, 0.80349011, 0.81272733, 0.81952092,
       0.8251123 , 0.83266813])


average_cost_M_2_s_3 = np.array([0.35773111, 0.32983733, 0.291976  , 0.259677  , 0.24197289,
        0.23059967, 0.2203616 , 0.21459233, 0.20886762, 0.20426767,
        0.19965252, 0.19525913])

# plt.plot(regret_x_label, optimal_cost_M_3_s_1,'--^', label = 's=1,optimal' )
plt.plot(regret_x_label, average_cost_M_3_s_1,'-^', label = 's=1,M=3' )
plt.plot(regret_x_label, average_cost_M_3_s_2,'-*', label = 's=2,M=3' )
plt.plot(regret_x_label, average_cost_M_3_s_3,'-o', label = 's=3,M=3' )
# plt.plot(regret_x_label, average_cost_M_2_s_1,'-^', label = 's=1,M=2' )
# plt.plot(regret_x_label, average_cost_M_2_s_2,'-*', label = 's=2,M=2' )
# plt.plot(regret_x_label, average_cost_M_2_s_3,'-o', label = 's=3,M=2' )

# plt.plot(regret_x_label, average_cost_M_1_s_2,'--^', label = 'M=1,s=2' )

plt.legend(ncol=1,framealpha = 0,fontsize = 14)
plt.xlabel('time',fontsize = 16)
plt.ylabel('average price',fontsize = 16)
plt.grid()
# fig.savefig('QoS_average_cost.eps', dpi = 600, format = 'eps')






































































