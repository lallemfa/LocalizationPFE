# -*- coding: utf-8 -*-
"""
Created on Tue May 29 11:32:51 2018

@author: fabrice.lallement
"""

import numpy as np
import time
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

from Environment import Environment



class Sonar:
    
    def __init__(self, environment, window_shape, number_of_ray = None):
        
        self.environment = environment
        
        self.cloud = np.reshape(environment.cloud, (1,-1,3))
        self.cloud = self.cloud[0, :, :].T
        
        nan_idx = np.argwhere(np.isnan(self.cloud))[:, 1]
        
        self.cloud = np.delete(self.cloud, nan_idx, 1)
        
        
        self.shape = window_shape
        
        if number_of_ray is None:
            number_of_ray = window_shape
        
        if type(number_of_ray) == int:
            self.x_number_ray = self.y_number_ray = number_of_ray
        elif type(number_of_ray) == list:
            self.x_number_ray = number_of_ray[0]
            self.y_number_ray = number_of_ray[1]
            
        self.grid_x, self.grid_y = np.mgrid[-self.shape[0]/2:self.shape[0]/2:self.x_number_ray*1j,
                                  -self.shape[1]/2:self.shape[1]/2:self.y_number_ray*1j]
        
        
    def generateView(self, window_center, window_rotation, flag_plot = False):
        
        window_center = np.hstack(( np.asarray(window_center), 0))
        
        psi = window_rotation
        
        R = np.array([[np.cos(psi), -np.sin(psi)],
                      [np.sin(psi),  np.cos(psi)]])
            
        X = np.reshape(self.grid_x, (1, -1))
        Y = np.reshape(self.grid_y, (1, -1))
        
        rot_grid = R@np.vstack((X, Y))
        
        rot_X = np.reshape(rot_grid[0, :], (self.x_number_ray, self.y_number_ray))
        rot_Y = np.reshape(rot_grid[1, :], (self.x_number_ray, self.y_number_ray))
        
        add_center = np.matlib.repmat(window_center, self.cloud.shape[1], 1).T
        
        pts = self.cloud - add_center
        
        grid_z = griddata(pts[0:2, :].T, pts[2, :].T, (rot_X, rot_Y), method='cubic')
        
        with_good_coord_view = np.dstack((rot_X, rot_Y, grid_z))
        
        with_original_grid_coord_view = np.dstack((self.grid_x, self.grid_y, grid_z))
        
        if flag_plot:
            self.plot_result(with_good_coord_view, window_center, R)
        
        return with_good_coord_view, with_original_grid_coord_view
    
    def plot_result(self, view, T, R):
        
        interp_pts = np.reshape(view, (1, -1, 3))[0, :, :].T + np.matlib.repmat(T, view.shape[1]*view.shape[0], 1).T
        
        fig 	= plt.figure("Superposition environment / interpolated data")
        
        ax 	= fig.add_subplot(111, projection = '3d')
#        ax.plot(origin_pts[0, ::20], origin_pts[1, ::20], origin_pts[2, ::20], 'g.')
        self.environment.plot(ax, 12)
#        ax.plot(origin_pts[0, :], origin_pts[1, :], origin_pts[2, :], 'g.')
        ax.plot(interp_pts[0, :], interp_pts[1, :], interp_pts[2, :], 'r.')
        
        plt.show()
    
if __name__ == '__main__':
    
    environment = Environment()
    
    sonar = Sonar(environment, [10, 20], [20,30])
    
    T = np.array([0, 0])
    
    start = time.time()
    view, _ = sonar.generateView(T, np.pi/10, flag_plot = True)
    print('Time elapsed {}'.format(time.time() - start))
    
    plt.figure("View")
    plt.imshow(view[:,:,2])
    plt.show()
    
    plt.figure("Env")
    plt.imshow(environment.cloud[:,:,2])
    plt.show()
    


