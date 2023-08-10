# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 14:39:25 2021

@author: xxx

This file is used to update $\pmb{\theta}$ and g_{n,s}(t)
"""

import numpy as np
from numpy.linalg import pinv
import math
import matplotlib.pyplot as plt

 

class my_SW_LinUCB_static(object):
    def __init__(self, theta, feature, dimension, lambdada):
        self.theta = theta
        self.feature = feature
        self.dimension = dimension
        self.lambdada = lambdada
        self.beta_t = np.sqrt(lambdada) 
        
        # some attributes
        self.cov = self.lambdada * np.identity(dimension)
        self.invcov = 1/self.lambdada * np.identity(dimension)
        # self.reward = np.dot(self.feature, self.theta) 
        self.b = np.zeros(dimension)
        self.hat_theta = np.zeros((self.dimension))
        self.aat = np.outer(self.feature, self.feature.T)
        self.ucb_s = []
        self.t_count = 0
        
    def update_information(self, feature, reward):
        self.aat = np.outer(feature, feature.T)
        self.cov = self.cov + self.aat
        self.b += np.dot(feature, reward)
        self.invcov = pinv(self.cov)
        self.hat_theta = np.inner(self.invcov, self.b)
        return self.hat_theta

    def compute_index(self, hat_theta, alpha, beta_t, feature, invcov, t_slot):
        invcov_a = np.inner(invcov, feature.T)
        self.ucb_s = np.dot(hat_theta, feature) + beta_t * alpha * np.sqrt(np.dot(feature, invcov_a)) * np.sqrt(2*np.log(t_slot))
        return self.ucb_s
        
    def obtain_reward(self, feature, theta):
        instant_reward = np.dot(feature, theta)
        return instant_reward
            
    def count_the_sensed_time(self):
        self.t_count += 1
    
    
    def re_init(self):
        """
        Re-init function to reinitialize the statistics while keeping the same hyperparameters
        """
        # self.t = 0
        self.hat_theta = np.zeros(self.dimension)
        self.cov = self.lambdada * np.identity(self.dimension)
        self.invcov = 1/self.lambdada * np.identity(self.dimension)
        self.b = np.zeros(self.dimension)
        self.aat = np.outer(self.feature, self.feature.T)
        self.ucb_s = []  
        self.t_count = 0
        
        
            
# we write a new function to compare with the previous version
# we add multiple algorithms for various scenarios. Since they are not related to my disseration, we do not give more explanations.
class my_SW_LinUCB_comparison(object):
    def __init__(self, theta, feature, dimension, lambdada):
        self.theta = theta
        self.feature = feature
        self.dimension = dimension
        self.lambdada = lambdada
        self.beta_t = np.sqrt(lambdada) 
        
        self.cov = self.lambdada * np.identity(dimension)
        self.cov_squared = self.lambdada * np.identity(dimension)
        self.invcov = 1/self.lambdada * np.identity(dimension)
        assert type(self.feature) == np.ndarray, "feature should be the np.ndarray" 
        self.aat = np.dot(self.feature, self.feature.T)
        self.b = np.zeros(dimension).reshape(dimension,1)
        self.hat_theta = np.zeros((self.dimension))
        # np.outer(self.feature, self.feature.T)
        self.ucb_s = []
        self.t_count = 0
        self.t_SW = 0

        self.a_tau = []
        self.reward_tau = []
        
        self.mu_ = 0.1
        self.cov_ = self.mu_ * np.identity(self.dimension)
        
    def update_information(self, feature, reward): # this is used to update $\pmb{\theta}$
        self.aat = np.dot(feature, feature.T)
        self.cov =  self.cov + self.aat
        self.b += feature * reward
        self.invcov = pinv(self.cov)
        self.hat_theta = np.dot(self.invcov, self.b)
        return self.hat_theta
    
    def compute_index(self, hat_theta, feature, invcov, t_slot): # compute the index
        invcov_a = np.dot(feature.T, invcov)
        self.ucb_s = np.dot(feature.T, hat_theta) + np.sqrt(np.dot(invcov_a, feature)) * np.sqrt(2*np.log(t_slot))
        return self.ucb_s
        

    # the following algorithms are not related to our paper, but for the subsuequent research.     
    
    
    def compute_index_UCB(self, hat_theta, alpha, beta_t, feature, invcov, t_slot):
        invcov_a = np.dot(feature.T, invcov)
        self.ucb_s = np.dot(feature.T, hat_theta) + np.sqrt(2*np.log(t_slot)) * np.sqrt(np.dot(invcov_a, feature))
        return self.ucb_s 
    
    def obtain_reward(self, feature, theta):
        instant_reward = np.dot(feature.T, theta)
        return instant_reward
    
    def count_the_sensed_time(self):
        self.t_count += 1        
        
    def update_information_weighted(self, feature, reward, alpha_t):
        self.aat = np.dot(feature, feature.T)
        self.cov = alpha_t * self.cov + self.aat + (1 - alpha_t) * alpha_t * np.identity(self.dimension)
        self.cov_squared = alpha_t ** 2 * self.cov + self.aat + (1 - alpha_t ** 2) * self.lambdada * np.identity(self.dimension)
        self.b = self.b * alpha_t + feature * reward
        self.invcov = pinv(self.cov)
        self.hat_theta = np.dot(self.invcov, self.b)
        return self.hat_theta

    def compute_index_weighted(self, hat_theta, alpha, beta_t, feature, invcov):
        invcov_temp = np.dot(np.dot(np.dot(np.dot(feature.T, self.invcov), self.cov_squared), self.invcov), feature)
        self.ucb_s = np.dot(feature.T, self.hat_theta) + beta_t * alpha * np.sqrt(invcov_temp)       
        return self.ucb_s



    def update_information_by_mu(self, feature, reward):
        self.aat = np.dot(feature, feature.T)
        self.cov = self.cov + self.aat
        self.cov_ = self.cov_ + self.aat
        self.b = self.b + feature * reward
        self.invcov = pinv(self.cov)
        self.invcov_ = pinv(self.cov_)
        self.hat_theta = np.dot(self.invcov, self.b)
        return self.hat_theta        
         
    def compute_index_mu(self, hat_theta, alpha, beta_t, feature):
        invcov_temp = np.dot(np.dot(np.dot(np.dot(feature.T, self.invcov), self.cov_),self.invcov),feature)
        self.ucb_s = np.dot(feature.T, self.hat_theta) + beta_t * alpha/np.sqrt(self.mu_) * np.sqrt(invcov_temp)
        return self.ucb_s


    def sliding_window_update_information(self, feature, reward, tau):
        if self.t_SW < tau:
            self.aat = np.dot(feature, feature.T)
            self.a_tau.append(feature)
            self.reward_tau.append(reward)
            self.cov =  self.cov + self.aat
            self.b += feature * reward
            self.invcov = pinv(self.cov)
            self.hat_theta = np.dot(self.invcov, self.b)
        else:
            self.aat = np.dot(feature, feature.T)
            act_delayed = self.a_tau.pop(0)
            aat_delayed = np.dot(act_delayed, act_delayed.T)
            rew_delayed = self.reward_tau.pop(0)
            self.cov = self.cov + self.aat - aat_delayed
            self.b = self.b + reward * feature - rew_delayed * act_delayed
            self.invcov = pinv(self.cov)
            self.a_tau.append(feature)
            self.reward_tau.append(reward)
            self.hat_theta = np.dot(self.invcov, self.b)
        self.t_SW += 1
        return self.hat_theta

    def sliding_window_update_information_add_weighted(self, feature, reward, tau):
        if self.t_SW < tau:
            self.aat = np.dot(feature, feature.T)
            self.a_tau.append(feature)
            self.reward_tau.append(reward)
            self.cov =  self.cov + self.aat
            self.b += feature * reward
            self.invcov = pinv(self.cov)
            self.hat_theta = np.dot(self.invcov, self.b)
        else:
            self.aat = np.dot(feature, feature.T)
            act_delayed = self.a_tau.pop(0)
            aat_delayed = np.dot(act_delayed, act_delayed.T)
            rew_delayed = self.reward_tau.pop(0)
            self.cov = 0.9 * self.cov + self.aat - 0.9 * aat_delayed
            self.b = self.b + 0.9 * reward * feature - 0.9 * rew_delayed * act_delayed
            self.invcov = pinv(self.cov)
            self.a_tau.append(feature)
            self.reward_tau.append(reward)
            self.hat_theta = np.dot(self.invcov, self.b)
        self.t_SW += 1
        return self.hat_theta        
        
    def weighted_update_information(self, feature, reward):
        self.aat = np.dot(feature, feature.T)
        # self.cov = 0.9 * self.cov + self.aat + (1 - 0.9) * self.lambdada * np.identity(self.dimension)
        self.cov = 1 * self.cov + self.aat
        self.b = 1 * self.b + reward * feature 
        self.invcov = pinv(self.cov)
        self.hat_theta = np.dot(self.invcov, self.b)
        return self.hat_theta

        
    
     
    def re_init(self):
        """
        Re-init function to reinitialize the statistics while keeping the same hyperparameters
        """
        # self.t = 0
        self.hat_theta = np.zeros(self.dimension).reshape(self.dimension,1)
        self.cov = self.lambdada * np.identity(self.dimension)
        self.cov_squared = self.lambdada * np.identity(self.dimension)
        self.invcov = 1/self.lambdada * np.identity(self.dimension)
        self.b = np.zeros(self.dimension).reshape(self.dimension,1)
        self.aat = np.dot(self.feature, self.feature.T)
        self.ucb_s = []  
        self.t_count = 0        

        self.t_SW = 0

        self.a_tau = []
        self.reward_tau = []
        self.cov_ = self.mu_ * np.identity(self.dimension)
        
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    