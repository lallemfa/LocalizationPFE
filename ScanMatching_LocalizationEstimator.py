from LocalizationEstimator import LocalizationEstimator
from StructMatch import StructMatch
from RobotState import RobotState
from DepthmapToImage import DepthmapToImage


from roblib import rotmat, kalman

import numpy as np

import time

import cv2

class ScanMatching_LocalizationEstimator(LocalizationEstimator):
    
    observation_time = 0
    
    computation_time = 0
    
    img_creation_time = 0
    
    observation = 0
    
    def __init__(self, dt, A, B, gammaINS, noise = None):
        
        super(ScanMatching_LocalizationEstimator, self).__init__(dt)
        
        self.A = A
        self.B = B
        
        self.gammaINS = gammaINS
        
        self.__converter = DepthmapToImage()
        
        self.__matcher = StructMatch()
    
        self.__noise = noise
    
    def compute_localization(self, i, previousState, insMeasures, previousInsMeasure, view):
        
        
        C, y, Gamma_obs = self.__generate_observations(i, previousState, view)
        
        
        start = time.time()
        
        correct_gravity = np.array([0, 0, 9.81])
        
# =============================================================================
#         Angular integration
# =============================================================================
        
        angular_velocity = insMeasures[3:6]
        
        previousAngles = previousState.orientation
        
        previousR = rotmat(previousAngles[0], previousAngles[1], previousAngles[2])
        
        angles = self.dt*angular_velocity + previousAngles
                
        R = rotmat(angles[0], angles[1], angles[2])
        
# =============================================================================
#         Velocities integration after correction of rotation and gravity
# =============================================================================
        
        velocities = self.dt*(previousR@previousInsMeasure[0:3] +
                              R@insMeasures[0:3] +
                              2*correct_gravity)/2 + previousState.linear_velocity
        
# =============================================================================
#         Kalman filter for computing pose
# =============================================================================
        
        pose, Gamma = kalman(previousState.pose, self.A, previousState.gamma,
                             self.dt*velocities, self.B, self.gammaINS,
                             y, Gamma_obs, C)
        
        estimatedState = RobotState(pose, angles, velocities, angular_velocity, Gamma, view = view)  
        
        ScanMatching_LocalizationEstimator.computation_time += time.time() - start
        
        return estimatedState
        
    
    def __generate_observations(self, i, previousState, data):
        
        C = None
        y = None
        Gamma_obs = None
        
        if len(data[0]) and len(data[1]):
            
            previous_view = data[0]
            view = data[1]
            previousPose = data[2]
            previousAngle = data[3]
            
            previous_view = previous_view[:,:,2]
            view = view[:,:,2]
            
            
    
            
    
            previous_view_img = cv2.imread("Environment_data/area_1/Images/RGB/RGB_sonar_view_state{}.png".format(i-50))
        
            if previous_view_img is None:
                print("Load failed")
                previous_view_img = self.__converter.img_gradient(previous_view.T, self.__noise)
    
    
            view_img = cv2.imread("Environment_data/area_1/Images/RGB/RGB_sonar_view_state{}.png".format(i))
        
            if view_img is None:
                print("Load failed")
                view_img = self.__converter.img_gradient(view.T, self.__noise)
            
            
            start = time.time()
            
            _, R, T = self.__matcher.matchImages(view_img, previous_view_img, False)
            
            ScanMatching_LocalizationEstimator.observation_time += time.time() - start
            
            if not((R is None) or (T is None)):
                
                ScanMatching_LocalizationEstimator.observation += 1
                
                rot = rotmat(-previousAngle)
                
                pose = previousPose + (rot@T).T
                
                y = np.array([pose[0, 0], pose[0, 1]])
                
                C = np.array([[1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0]])
                
                Gamma_obs = np.eye(2)
        
        return C, y, Gamma_obs
    
    
    
    
    