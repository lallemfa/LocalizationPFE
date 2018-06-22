import numpy as np
import matplotlib.pyplot as plt

import time
import cv2

from Environment import Environment


class DepthmapToImage:

    def __init__(self):
        
        self.__depthRange = None
        self.__gradMax = None
    		
        
    def init_with_environment(self, environment, precision = None):
        if precision is None:
            cloud = environment.cloud[:, :, 2]
        else:
            cloud = environment.reduceResolution(precision)
            cloud = cloud[:, :, 2]
            
        [Gx, Gy]    = np.gradient(cloud)
        
        norm_grad = np.sqrt((Gx**2) + (Gy**2))
        self.__gradMax = np.nanmax(norm_grad)
        norm_grad /= self.__gradMax
        norm_grad *= 255
        

        Z = cloud
        self.__depthRange = [np.nanmin(Z), np.nanmax(Z)]
        Z -= self.__depthRange[0]
        Z /= (self.__depthRange[1] - self.__depthRange[0])
        Z *= 255
        
        img_grad = np.dstack((0*norm_grad, norm_grad))
        img_grad = np.dstack((img_grad, Z))
#        
#        for i in range(len(Gx)):
#            for j in range(len(Gx[0])):
#                
#                img_grad[i, j, 0] = (np.arctan2(Gx[i, j], Gy[i, j]) + np.pi/2) * (255/np.pi)
        
        return img_grad.astype(np.uint8)




    def img_gradient(self, depth, stdNoise = None):
        
        if not stdNoise is None:
            depth += stdNoise * np.random.randn(depth.shape[0], depth.shape[1])
        
        [Gx, Gy] = np.gradient(depth)
        
        norm_grad = np.sqrt((Gx**2) + (Gy**2))
        norm_grad /= self.__gradMax if not self.__gradMax is None else np.nanmax(norm_grad)
        norm_grad[norm_grad>1] = 1
        norm_grad *= 255
        

        Z =  depth
        if not self.__depthRange is None:
            Z -= self.__depthRange[0]
            Z /= (self.__depthRange[1]-self.__depthRange[0])
            Z[Z>1] = 1
        else:
            Z -= np.nanmin(Z)
            Z /= np.nanmax(Z)
        
        Z *= 255
        
        img_grad = np.dstack((0*norm_grad, norm_grad))
        img_grad = np.dstack((img_grad, Z))
#        
#        for i in range(len(Gx)):
#            for j in range(len(Gx[0])):
#                
#                img_grad[i, j, 0] = (np.arctan2(Gx[i, j], Gy[i, j]) + np.pi/2) * (255/np.pi)

        return img_grad.astype(np.uint8)

    
    
