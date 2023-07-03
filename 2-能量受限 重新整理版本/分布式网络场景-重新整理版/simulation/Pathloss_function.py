# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 19:12:15 2022

@author: xxx
"""

import random
import numpy as np

def Rayleigh_pathloss_function(transmission_distance, Tx_Num, channel_noise):
    # channel noise is in the form of num
    PathLoss_Model_Parameter = np.sqrt( 10**(-3.53)/( transmission_distance**(3.76)) )
    Channel_parameter_vector = []
    for i in range( Tx_Num ):
        Channel_parameter_vector.append( PathLoss_Model_Parameter * 1/np.sqrt(2) * ( np.random.normal() + complex(0, 1) * np.random.normal() ) )
        # 其他场景直接求信噪比比较好 或者信道系数
    Channel_parameter_norm = np.linalg.norm( Channel_parameter_vector )
    Channel_gain_noise_ratio = Channel_parameter_norm**2/channel_noise
    return Channel_gain_noise_ratio

def Rician_pathloss_function(transmission_distance, Tx_Num, channel_noise, K_factor):
    # channel noise is in the form of num
    PathLoss_Model_Parameter = np.sqrt( 10**(-3.53)/( transmission_distance**(3.76)) )
    Channel_parameter_vector = []
    for i in range( Tx_Num ):
        Channel_parameter_vector.append( PathLoss_Model_Parameter * ( np.sqrt( K_factor/(K_factor+1) ) * ( np.random.normal() + complex(0, 1) * np.random.normal() ) + np.sqrt( 1/(K_factor+1) ) * np.random.normal()   )  )
        # 其他场景直接求信噪比比较好 或者信道系数
    Channel_parameter_norm = np.linalg.norm( Channel_parameter_vector )
    Channel_gain_noise_ratio = Channel_parameter_norm**2/channel_noise
    return Channel_gain_noise_ratio


def transmission_rate(transmit_power, channel_gain_noise_ratio, bandwidth):
    achievable_rate = bandwidth * np.log( 1 + transmit_power *  channel_gain_noise_ratio)
    return achievable_rate


def Transmit_Power_energy_minimum(Channel_gain_noise_ratio, transmission_rate, bandwidth):
    Power_minimum = ( 2**(transmission_rate/bandwidth) - 1 )/Channel_gain_noise_ratio
    return Power_minimum

channel_noise = -104 # dBm
channel_noise = 10**(channel_noise/10) * 10**(-3)


Channel_gain_noise_ratio = Rayleigh_pathloss_function(100, 4, channel_noise )
Channel_gain_noise_ratio_Rician = Rician_pathloss_function(100, 4, channel_noise, 1 )

print('the Rayleigh fading parameter is', Channel_gain_noise_ratio)
print('the Rician fading parameter is', Channel_gain_noise_ratio_Rician)


Transmit_Power_Minimum = Transmit_Power_energy_minimum(  Channel_gain_noise_ratio, 5, 1   )
Transmit_Power_Minimum_Rician = Transmit_Power_energy_minimum(  Channel_gain_noise_ratio_Rician, 5, 1   )

print(Transmit_Power_Minimum)
print(Transmit_Power_Minimum_Rician)

achievable_rate_Rayleigh = transmission_rate( 10e-03, Channel_gain_noise_ratio, 1 )
achievable_rate_Rician = transmission_rate( 10e-03, Channel_gain_noise_ratio_Rician, 1 )

print('the achievable Rayleigh transmission rate', achievable_rate_Rayleigh)
print('the achievable Rician transmission rate', achievable_rate_Rician)













































