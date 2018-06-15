# -*- coding: utf-8 -*-
"""
Created on Wed May 23 15:05:24 2018

@author: fabrice.lallement
"""

import numpy as np
import matplotlib.pyplot as plt

from RobotState import RobotState
from roblib import rotmat


class INS:

    def __init__(self, linear_bias = [0, 0, 0], \
                 linear_white_noise_standard_deviation = [0, 0, 0], \
                 linear_bias_instability_standard_deviation = [0, 0, 0], \
                 angular_bias = [0, 0, 0], \
                 angular_white_noise_standard_deviation = [0, 0, 0], \
                 angular_bias_instability_standard_deviation = [0, 0, 0], \
                 flag_z_axis = 1):
        
        self.linear_bias                                    = linear_bias
        self.linear_white_noise_standard_deviation          = linear_white_noise_standard_deviation
        self.linear_bias_instability_standard_deviation     = linear_bias_instability_standard_deviation
        
        self.angular_bias                                   = angular_bias
        self.angular_white_noise_standard_deviation         = angular_white_noise_standard_deviation
        self.angular_bias_instability_standard_deviation    = angular_bias_instability_standard_deviation
        
        self.noise_history = np.array([])
        self.measure_history = np.array([])
        
        self.previous_random_walk = np.zeros(6)
        
        if flag_z_axis != 1 and flag_z_axis != -1:
            self.coeff_z = 1
        else:
            self.coeff_z = flag_z_axis
        
        self.linear_acceleration = np.zeros(3)
        
        self.angular_rate   = []
    
        self.gamma_measure = 0.01*np.eye(3)
        
    def getCovarianceOfMeasures(self):
        return self.gamma_measure
    
    def generateMeasures(self, h, state, state_minus_1, state_minus_2):
    
        add_gravity = self.coeff_z*np.array([0.0, 0.0, -9.81])
                
        self.linear_acceleration = (state.pose - 2.0*state_minus_1.pose + state_minus_2.pose)/(h**2.0)

        self.angular_rate = (state.orientation - state_minus_1.orientation)/h
            
        phi, theta, psi = state.orientation
        
        R = rotmat(phi, theta, psi)
        
        self.linear_acceleration += add_gravity
        
        measure = np.hstack((self.linear_acceleration, self.angular_rate))
        
        noise = self.generateNoise()
        
        measure = measure + noise
        
        measure[0:3] = (R.T)@measure[0:3]
        
        self.measure_history = np.vstack((self.measure_history, measure)) if len(self.measure_history) else measure
        
        return measure
    
    def generateNoise(self):
        
        constant_noise = np.asarray(self.linear_bias + self.angular_bias)
        
        white_noise = np.array([np.random.normal(0, self.linear_white_noise_standard_deviation[0]),
                                np.random.normal(0, self.linear_white_noise_standard_deviation[1]),
                                np.random.normal(0, self.linear_white_noise_standard_deviation[2]),
                                np.random.normal(0, self.angular_white_noise_standard_deviation[0]),
                                np.random.normal(0, self.angular_white_noise_standard_deviation[1]),
                                np.random.normal(0, self.angular_white_noise_standard_deviation[2])])
    
        instability = np.array([np.random.normal(0, self.linear_bias_instability_standard_deviation[0]),
                                np.random.normal(0, self.linear_bias_instability_standard_deviation[1]),
                                np.random.normal(0, self.linear_bias_instability_standard_deviation[2]),
                                np.random.normal(0, self.angular_bias_instability_standard_deviation[0]),
                                np.random.normal(0, self.angular_bias_instability_standard_deviation[1]),
                                np.random.normal(0, self.angular_bias_instability_standard_deviation[2])])
    
        random_walk_noise = instability + self.previous_random_walk
        
        self.previous_random_walk = random_walk_noise
        
        noise = np.vstack((constant_noise, white_noise, random_walk_noise))
        
        noise = np.sum(noise, axis = 0)
                
        self.noise_history = np.vstack((self.noise_history, noise)) if len(self.noise_history) else noise
        
        return noise
    
    def integrateHistory(self, dt):
        
        
        state = RobotState()
        
        states = [state]
        
        for i in range(1, len(ins.measure_history)):
            previous_measure = self.measure_history[i-1, :]
            measure = self.measure_history[i, :]
            
            correct_gravity = np.array([0, 0, 9.81])
        
            previous_state = states[-1]
            
            angles = dt*(previous_measure[3:6] + measure[3:6])/2 + previous_state.orientation
            
            R = rotmat(angles[0], angles[1], angles[2])
            
            velocities = dt*(previous_measure[0:3] + R@measure[0:3] + 2*correct_gravity)/2 + previous_state.linear_velocity
            
            pose = dt*velocities + previous_state.pose
            
            estimated_state = RobotState(pose, angles, velocities)
        
            states.append(estimated_state)
        
        return states
    
    def plot(self):
        
        plt.figure("Linear noises")
        plt.plot(self.noise_history[:, 0], label='X Axis')
        plt.plot(self.noise_history[:, 1], label='Y Axis')
        plt.plot(self.noise_history[:, 2], label='Z Axis')
        plt.legend()
        plt.show()
        
        plt.figure("Angular noises")
        plt.plot(self.noise_history[:, 3], label='Roll')
        plt.plot(self.noise_history[:, 4], label='Pitch')
        plt.plot(self.noise_history[:, 5], label='Yaw')
        plt.legend()
        plt.show()
        
        plt.figure("Linear measures")
        plt.plot(self.measure_history[:, 0], label='X Axis')
        plt.plot(self.measure_history[:, 1], label='Y Axis')
        plt.plot(self.measure_history[:, 2], label='Z Axis')
        plt.legend()
        plt.show()
        
        plt.figure("Angular measures")
        plt.plot(self.measure_history[:, 3], label='Roll')
        plt.plot(self.measure_history[:, 4], label='Pitch')
        plt.plot(self.measure_history[:, 5], label='Yaw')
        plt.legend()
        plt.show()
    
    def __str__(self):
        return "\tINS\n\n" + \
                "Linear bias                -> {}\n".format(self.linear_bias) + \
                "Linear white noise std     -> {}\n".format(self.linear_white_noise_standard_deviation) + \
                "Linear instability std     -> {}\n".format(self.linear_bias_instability_standard_deviation) + \
                "\n" + \
                "Angular bias               -> {}\n".format(self.angular_bias) + \
                "Angular white noise std    -> {}\n".format(self.angular_white_noise_standard_deviation) + \
                "Angular instability std    -> {}\n".format(self.angular_bias_instability_standard_deviation)
    
    
if __name__ == '__main__':
    
    plt.close("all")
    
    ins = INS(linear_bias = [1e-4, 0, 0], \
             linear_white_noise_standard_deviation = [0, 1e-4, 0], \
             linear_bias_instability_standard_deviation = [0, 0, 1e-4], \
             angular_bias = [0, 0, 0], \
             angular_white_noise_standard_deviation = [0, 0, 0], \
             angular_bias_instability_standard_deviation = [0, 0, 0])
    
    print(ins)
    
    state = RobotState()
    
    for i in range(1000):
        ins.generateMeasures(0.1, state, state, state)
    
    ins.plot()
    
    states = ins.integrateHistory(0.1)
    
    estimated_trajectory = []
    for state in states:
        estimated_trajectory = np.vstack((estimated_trajectory, state.pose)) if len(estimated_trajectory) else state.pose
    
    fig 	= plt.figure("Run plot")
    ax 	= fig.add_subplot(111, projection = '3d')
    
    ax.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], estimated_trajectory[:, 2], '-.', label='Estimated trajectory')
      
    plt.show()
    
    
    
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(estimated_trajectory[:, 0])
    ax2.plot(estimated_trajectory[:, 1])
    ax3.plot(estimated_trajectory[:, 2])
    
    
    