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

fig = plt.figure()
X = [1,2,3,4,5,6,7,8,9,10]

# --------------------------------------------------------------------------------------------- #
regret_x_label = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000] 
# --------------------------------------------------------------------------------------------- #

Throuhgput_y_label_s_1_proposed = [ 93.79445692, 191.45611162, 290.86332201, 390.0361554 ,
       490.03841736, 590.1107985 , 690.61971055, 790.70921247,
       891.83824983, 991.77609913]

Energy_cost_y_label_s_1_proposed = [ 36.38628521,  75.39576864, 114.79510454, 154.93162103,
       195.20767451, 235.3443102 , 275.97502295, 316.24972677,
       357.013635  , 397.41138723]

Energy_cost_y_label_s_1_proposed = [ element * -1 for element in Energy_cost_y_label_s_1_proposed ]
# --------------------------------------------------------------------------------------------- #

# --------------------------------------------------------------------------------------------- #
Throuhgput_y_label_s_2_proposed = [ 78.56009988, 165.25828001, 255.65763853, 347.00745892,
       440.7912805 , 534.59376782, 630.80575733, 727.04159787,
       822.67534506, 919.26020756]
Energy_cost_y_label_s_2_proposed = [ 26.25473892,  58.14055863,  92.76322527, 128.51231301,
       164.98516396, 201.77119338, 239.90466777, 277.69213613,
       316.14987273, 354.83234188]

Energy_cost_y_label_s_2_proposed = [ element * -1 for element in Energy_cost_y_label_s_2_proposed ]
# --------------------------------------------------------------------------------------------- #

# --------------------------------------------------------------------------------------------- #

Throuhgput_y_label_s_1_random = [ 62.71977764, 127.35398026, 191.52150868, 255.3407254 ,
       319.02057001, 382.63479421, 447.00424286, 512.181627  ,
       575.75752845, 639.16856437] 

Energy_cost_y_label_s_1_random = [ 16.86253509,  33.94339347,  50.85060467,  67.39886669,
        84.55213557, 101.15738998, 118.47058412, 136.1293297 ,
       152.78828643, 169.37099432]

Energy_cost_y_label_s_1_random = [ element * -1 for element in Energy_cost_y_label_s_1_random ]
# --------------------------------------------------------------------------------------------- #


# --------------------------------------------------------------------------------------------- #

Throuhgput_y_label_s_1_fixed = [ 49.34227729,  97.70371418, 145.04427636, 191.48406675,
       240.30589811, 288.06682047, 336.54836026, 383.9890082 ,
       431.16943318, 479.1905785 ]

Energy_cost_y_label_s_1_fixed = [ 3.21345896,  6.34774802,  9.42818201, 12.426943  , 15.52307419,
       18.48505799, 21.51928866, 24.63668644, 27.6440586 , 30.75173696]

Energy_cost_y_label_s_1_fixed = [ element * -1 for element in Energy_cost_y_label_s_1_fixed ]
# --------------------------------------------------------------------------------------------- #



X_values = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]


# plt.plot(regret_x_label, Throuhgput_y_label_s_1_proposed)
# plt.plot(regret_x_label, Throuhgput_y_label_s_2_proposed)

# plt.plot(regret_x_label, Energy_cost_y_label_s_1_proposed)
# plt.plot(regret_x_label, Energy_cost_y_label_s_2_proposed)


l3= plt.bar(  [ x + 0.10 for x in X ],  Throuhgput_y_label_s_1_proposed,   align='center', facecolor='#53fca1',     label = 'throughput,proposed,s=1', width = 0.4)
l1= plt.bar(  [ x - 0.10 for x in X ],  Throuhgput_y_label_s_2_proposed,   align='center', facecolor='#9999ff',     label = 'throughput,proposed,s=2', width = 0.4)
l6= plt.bar(  [ x + 0.20 for x in X ],  Throuhgput_y_label_s_1_random,   align='center', facecolor='#ffcce9',      label = 'throughput,random,s=1', width = 0.4) # o
l8= plt.bar(  [ x - 0.20 for x in X ],  Throuhgput_y_label_s_1_fixed,   align='center', facecolor='#FF6347',      label = 'throughput,fixed,s=1', width = 0.4) # o




l4= plt.bar(  [ x + 0.10 for x in X ],  Energy_cost_y_label_s_1_proposed,   align='center', facecolor='#ff63e9',      label = 'energy,proposed,s=1', width = 0.4) # o
l2= plt.bar(  [ x - 0.10 for x in X ],  Energy_cost_y_label_s_2_proposed,   align='center', facecolor='#ff9999',     label = 'energy,proposed,s=2', width = 0.4) # o
l5= plt.bar(  [ x + 0.20 for x in X ],  Energy_cost_y_label_s_1_random,   align='center', facecolor='#DCDCDC',     label = 'energy,random,s=1', width = 0.4) # o
l7= plt.bar(  [ x - 0.20 for x in X ],  Energy_cost_y_label_s_1_fixed,   align='center', facecolor='#008080',     label = 'energy,fixed,s=1', width = 0.4) # o



plt.xticks(X, X_values)
 
plt.grid(True)
plt.ylim([-450,1300 ])
plt.legend( ncol=1,  fontsize = 9.5, framealpha = 0 )
# fig.savefig('energy_throughput.eps', dpi = 600, format = 'eps')

 
 

























