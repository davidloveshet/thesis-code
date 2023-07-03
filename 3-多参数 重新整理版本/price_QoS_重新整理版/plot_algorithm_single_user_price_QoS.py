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
feature_generation = np.array([   [1], [0]  ])
T_simulation = 10000
regret_x_label = [ round(0.03 * T_simulation), round(0.05 * T_simulation), round(0.1 * T_simulation), round(0.2 * T_simulation), round(0.3 *T_simulation), round(0.4 *T_simulation), round(0.5 *T_simulation), round(0.6 *T_simulation), round(0.7 *T_simulation), round(0.8 *T_simulation), round(0.9 *T_simulation),  T_simulation  ]

# cumulative_utility_regret_y_label_s_1_M_2 = np.array([ 63.53333333,  86.26666667, 124.53333333, 178.76666667,
#        220.63333333, 247.3       , 268.63333333, 290.26666667,
#        300.4       , 312.33333333, 327.73333333, 339.23333333])


cumulative_utility_regret_y_label_s_1_M_3 = np.array([ 65.6       ,  87.26666667, 118.13333333, 160.63333333,
       193.33333333, 222.86666667, 244.        , 264.33333333,
       278.56666667, 294.43333333, 303.1       , 314.26666667])


# cumulative_utility_regret_y_label_s_2_M_2 = np.array([ 53.58837667,  79.74901   , 124.57417333, 181.05570333,
#        235.08692333, 267.40069   , 303.21664333, 331.60161333,
#        361.16582667, 369.20221667, 386.36318   , 407.79912   ])


cumulative_utility_regret_y_label_s_2_M_3 = np.array([ 61.66117   ,  85.765655  , 129.92327   , 186.72710333,
       228.04430667, 251.59313333, 278.21651   , 300.15679   ,
       319.29637833, 336.21319   , 353.35081833, 367.74819333])

# cumulative_utility_regret_y_label_s_3_M_2 = np.array([ 70.5855504,  96.3881008, 139.9039776, 194.890902 , 243.082124 ,
#        272.79918  , 299.9386208, 329.1814484, 349.0415304, 371.0823088,
#        388.296894 , 390.3465412])


cumulative_utility_regret_y_label_s_3_M_3 = np.array([ 65.2705154,  88.4516104, 135.6085808, 203.2318536, 250.4911178,
       286.8714128, 314.5704054, 337.4663034, 349.7806856, 361.8923658,
       370.7108456, 382.8116836])


cumulative_utility_regret_y_label_fixed = np.array([ 285.76666667,  477.        ,  954.26666667, 1898.9       ,
       2852.36666667, 3800.76666667, 4742.03333333, 5689.03333333,
       6636.5       , 7589.76666667, 8538.43333333, 9491.4       ])

cumulative_utility_regret_y_label_random = np.array([ 134.73333333,  219.1       ,  437.7       ,  872.9       ,
       1316.33333333, 1752.43333333, 2188.46666667, 2629.93333333,
       3063.96666667, 3498.2       , 3926.06666667, 4356.23333333])


cumulative_utility_regret_y_label_greedy = np.array([ 66.96666667,  91.66666667, 140.33333333, 223.03333333,
       303.53333333, 378.1       , 441.83333333, 507.6       ,
       565.86666667, 638.36666667, 703.46666667, 768.16666667])



# plt.plot(regret_x_label, cumulative_utility_regret_y_label_M_1_K_1,'-o', label = 'M=1,K=1' )
# plt.plot(regret_x_label, cumulative_utility_regret_y_label_M_2_K_1,'-.', label = 'M=2,K=1' )
# plt.plot(regret_x_label, cumulative_utility_regret_y_label_M_2_K_2,'-^', label = 'M=2,K=2' )
# plt.plot(regret_x_label, cumulative_utility_regret_y_label_M_3_K_1,'--o', label = 'M=3,K=1' )
# plt.plot(regret_x_label, cumulative_utility_regret_y_label_M_3_K_2,'--.', label = 'M=3,K=2' )
plt.plot(regret_x_label, cumulative_utility_regret_y_label_s_1_M_3,'-^', label = 's=1,proposed' )
plt.plot(regret_x_label, cumulative_utility_regret_y_label_s_2_M_3,'-*', label = 's=2,proposed' )
plt.plot(regret_x_label, cumulative_utility_regret_y_label_s_3_M_3,'-o', label = 's=3,proposed' )
plt.plot(regret_x_label[0:3], cumulative_utility_regret_y_label_fixed[0:3],'-x', label = 's=1,fixed' )
plt.plot(regret_x_label[0:4], cumulative_utility_regret_y_label_random[0:4],'--^', label = 's=1,random' )
plt.plot(regret_x_label, cumulative_utility_regret_y_label_greedy,'--o', label = 's=1,greedy' )
# plt.plot(regret_x_label, cumulative_utility_regret_y_label_s_1_M_2,'--^', label = 's=1,M=2' )
# plt.plot(regret_x_label, cumulative_utility_regret_y_label_s_2_M_2,'--*', label = 's=2,M=2' )
# plt.plot(regret_x_label, cumulative_utility_regret_y_label_s_3_M_2,'--o', label = 's=3,M=2' )
plt.legend(ncol=2,framealpha = 0,fontsize=11)
plt.xlabel('time',fontsize=16)
plt.ylabel('utility loss',fontsize=16)
# plt.title('s=2',fontsize=16)
# plt.xlim([-10,12000])
plt.grid()
fig.savefig('regret_QoS.eps', dpi = 600, format = 'eps')

 

# In[] average throughput 
import numpy as np
import random 
import matplotlib
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
feature_generation = np.array([   [1], [0]  ])
T_simulation = 10000
regret_x_label = [ round(0.03 * T_simulation), round(0.05 * T_simulation),  round(0.1 * T_simulation), round(0.2 * T_simulation), round(0.3 *T_simulation), round(0.4 *T_simulation), round(0.5 *T_simulation), round(0.6 *T_simulation), round(0.7 *T_simulation), round(0.8 *T_simulation), round(0.9 *T_simulation),  T_simulation  ]

optimal_throughput_M_3_s_3 = 1.97 * np.ones(len(regret_x_label)) # M bits 
average_throughput_M_3_s_3 = np.array([1.85111111, 1.87246667, 1.87963333, 1.88835   , 1.89765556,
       1.90531667, 1.91195333, 1.91743889, 1.92312381, 1.927675  ,
       1.93151852, 1.93399   ])

optimal_throughput_M_3_s_2 = 2.29 * np.ones(len(regret_x_label)) # M bits 
average_throughput_M_3_s_2 = np.array([2.09977778, 2.1406    , 2.19123333, 2.23098333, 2.2494    ,
       2.26163333, 2.26708667, 2.27169444, 2.27564762, 2.27884167,
       2.28062963, 2.28206333])

optimal_throughput_M_3_s_1 = 2.4 * np.ones(len(regret_x_label)) # M bits 
average_throughput_M_3_s_1 = np.array([2.18133333, 2.22546667, 2.28186667, 2.31968333, 2.33555556,
       2.34428333, 2.3512    , 2.35594444, 2.36020476, 2.36319583,
       2.36632222, 2.36857333])


average_throughput_fixed = np.array([1.44744444, 1.446     , 1.44573333, 1.45055   , 1.44921111,
       1.44980833, 1.45159333, 1.45182778, 1.45192857, 1.45127917,
       1.45128519, 1.45086   ])


average_throughput_random = np.array([1.95088889, 1.9618    , 1.9623    , 1.96355   , 1.96122222,
       1.96189167, 1.96230667, 1.96167778, 1.96229048, 1.962725  ,
       1.96377037, 1.96437667])


average_throughput_greedy = np.array([2.17677778, 2.21666667, 2.25966667, 2.28848333, 2.29882222,
       2.305475  , 2.31163333, 2.3154    , 2.3191619 , 2.32020417,
       2.32183704, 2.32318333])



# optimal_throughput_M_2_s_1 = 1.65 * np.ones(len(regret_x_label)) # M bits 
# average_throughput_M_2_s_1 = np.array([1.43822222, 1.47746667, 1.52546667, 1.56061667, 1.57645556,
#        1.588175  , 1.59627333, 1.60162222, 1.60708571, 1.61095833,
#        1.61358519, 1.61607667])

# optimal_throughput_M_2_s_2 = 1.5 * np.ones(len(regret_x_label)) # M bits 
# average_throughput_M_2_s_2 = np.array([1.36022222, 1.38386667, 1.42233333, 1.46243333, 1.48364444,
#        1.50021667, 1.51048   , 1.51916667, 1.52530476, 1.53229167,
#        1.53688889, 1.54102667])

# optimal_throughput_M_2_s_3 = 1.21 * np.ones(len(regret_x_label)) # M bits 
# average_throughput_M_2_s_3 = np.array([1.20022222, 1.21706667, 1.23333333, 1.24423333, 1.24253333,
#        1.24403333, 1.24165333, 1.24061111, 1.23977143, 1.2384    ,
#        1.23680741, 1.23655333])


# average_throughput_M_1_s_2 = np.array([0.00669867, 0.0068248 , 0.007062  , 0.0072944 , 0.007436  ,
#        0.0075346 , 0.00758912, 0.00764187, 0.00767589, 0.0077098 ,
#        0.00772973, 0.00775192])


# plt.plot(regret_x_label, average_throughput_M_1_K_1,'-o', label = 'M=1,K=1' )
# plt.plot(regret_x_label, average_throughput_M_2_K_1,'-.', label = 'M=2,K=1' )
# plt.plot(regret_x_label, average_throughput_M_2_K_2,'-^', label = 'M=2,K=2' )
# plt.plot(regret_x_label, average_throughput_M_3_K_1,'--o', label = 'M=3,K=1' )
# plt.plot(regret_x_label, average_throughput_M_3_K_2,'--.', label = 'M=3,K=2' )
plt.plot(regret_x_label, optimal_throughput_M_3_s_1,'-', label = 's=1,optimal' )
plt.plot(regret_x_label, optimal_throughput_M_3_s_2,'-', label = 's=2,optimal' )
plt.plot(regret_x_label, optimal_throughput_M_3_s_3,'-', label = 's=3,optimal' )
plt.plot(regret_x_label, average_throughput_M_3_s_1,'-^', label = 's=1,proposed' )
plt.plot(regret_x_label, average_throughput_M_3_s_2,'-*', label = 's=2,proposed' )
plt.plot(regret_x_label, average_throughput_M_3_s_3,'-o', label = 's=3,proposed' )
plt.plot(regret_x_label, average_throughput_fixed,'-x', label = 's=1,fixed' )
plt.plot(regret_x_label, average_throughput_random,'--^', label = 's=1,random' )
plt.plot(regret_x_label, average_throughput_greedy,'--o', label = 's=1,greedy' )
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

plt.legend(ncol=2,framealpha = 0,fontsize = 11.1)
plt.xlabel('time',fontsize = 16)
plt.ylabel('average throughput (Mbps)',fontsize = 16)
# plt.title('s=1',fontsize = 14)
# plt.ylim([0,2.6])
# plt.xlim([-20, 16000])
plt.grid()
fig.savefig('QoS_average_throughput.eps', dpi = 600, format = 'eps')

################## 



# In[] average cost 
import numpy as np
import random 
import matplotlib
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
 

regret_x_label = [ round(0.03 * T_simulation), round(0.05 * T_simulation),  round(0.1 * T_simulation), round(0.2 * T_simulation), round(0.3 *T_simulation), round(0.4 *T_simulation), round(0.5 *T_simulation), round(0.6 *T_simulation), round(0.7 *T_simulation), round(0.8 *T_simulation), round(0.9 *T_simulation),  T_simulation  ]

average_cost_M_3_s_1 = np.array([1.28364889, 1.33679467, 1.41073867, 1.46088183, 1.48147356,
       1.492029  , 1.5000768 , 1.50535889, 1.51010652, 1.51346429,
       1.5167833 , 1.5195906 ])

average_cost_M_3_s_2 = np.array([1.05481111, 1.084166  , 1.12423767, 1.14250517, 1.14922911,
       1.14877333, 1.14351673, 1.14106728, 1.14016038, 1.13954775,
       1.13661926, 1.13328517])

 
average_cost_M_3_s_3 = np.array([0.60524222, 0.584946  , 0.54961767, 0.52362383, 0.51376011,
       0.5087865 , 0.50599913, 0.50438439, 0.50323629, 0.50279758,
       0.50228385, 0.5016635 ])

average_cost_M_3_fixed = np.array([0.22845222, 0.22808   , 0.22781067, 0.22863883, 0.228461  ,
       0.22853642, 0.22876527, 0.22876106, 0.2287969 , 0.22869612,
       0.22867359, 0.22860093])

average_cost_M_3_random = np.array([0.93318667, 0.93761133, 0.94177367, 0.9429485 , 0.94304022,
       0.94350425, 0.94340453, 0.94282611, 0.94309567, 0.94341425,
       0.94342948, 0.94376563])


average_cost_M_3_greedy = np.array([1.26100556, 1.31625467, 1.377684  , 1.41476583, 1.42818244,
       1.43697108, 1.44424367, 1.44875533, 1.45286476, 1.45468537,
       1.45692511, 1.4587801 ])


# average_cost_M_2_s_1 = np.array([0.82826889, 0.88163867, 0.95902533, 1.01647317, 1.04802878,
#        1.07275075, 1.092747  , 1.10719278, 1.12034676, 1.13128233,
#        1.1372057 , 1.1424857 ])

# average_cost_M_2_s_2 = np.array([0.65169556, 0.671452  , 0.68972467, 0.718495  , 0.75415556,
#        0.77457083, 0.79035707, 0.80349011, 0.81272733, 0.81952092,
#        0.8251123 , 0.83266813])


# average_cost_M_2_s_3 = np.array([0.35773111, 0.32983733, 0.291976  , 0.259677  , 0.24197289,
#         0.23059967, 0.2203616 , 0.21459233, 0.20886762, 0.20426767,
#         0.19965252, 0.19525913])

# plt.plot(regret_x_label, optimal_cost_M_3_s_1,'--^', label = 's=1,optimal' )
plt.plot(regret_x_label, average_cost_M_3_s_1,'-^', label = 's=1,proposed' )
plt.plot(regret_x_label, average_cost_M_3_s_2,'-*', label = 's=2,proposed' )
plt.plot(regret_x_label, average_cost_M_3_s_3,'-o', label = 's=3,proposed' )
plt.plot(regret_x_label, average_cost_M_3_fixed,'-x', label = 's=1,fixed' )
plt.plot(regret_x_label, average_cost_M_3_random,'--^', label = 's=1,random' )
plt.plot(regret_x_label, average_cost_M_3_greedy,'--o', label = 's=1,greedy' )
# plt.plot(regret_x_label, average_cost_M_2_s_1,'-^', label = 's=1,M=2' )
# plt.plot(regret_x_label, average_cost_M_2_s_2,'-*', label = 's=2,M=2' )
# plt.plot(regret_x_label, average_cost_M_2_s_3,'-o', label = 's=3,M=2' )

# plt.plot(regret_x_label, average_cost_M_1_s_2,'--^', label = 'M=1,s=2' )

plt.legend(ncol=3,framealpha = 0,fontsize = 9.5)
plt.xlabel('time',fontsize = 16)
plt.ylabel('average price',fontsize = 16)
plt.ylim([0.05, 1.6])
plt.grid()
fig.savefig('QoS_average_cost.eps', dpi = 600, format = 'eps')























































 
