from LocalizationEstimator import LocalizationEstimator
from StructMatch import StructMatch
from Environment import Environment
from RobotState import RobotState
from DepthmapToImage import DepthmapToImage

from roblib import rotmat, kalman

import time
import numpy as np
import cv2

class TBN_SURF_LocalizationEstimator(LocalizationEstimator):
    
    match_time = 0
    
    def __init__(self, dt, A, B, gammaINS, environment, precision = 2, noise = None):
        
        super(TBN_SURF_LocalizationEstimator, self).__init__(dt)
        
        self.A = A
        self.B = B
        
        self.gammaINS = gammaINS
        
        
        self.__matcher = StructMatch()
        self.__converter = DepthmapToImage()
        
        self.__noise = noise
        
        self.__environment  = environment
        
        self.__precision = precision
        
        self.__fullMapImage = cv2.imread("Environment_data/area_1/Images/RGB/map_rgb_{}m.png".format(precision))
        
        if self.__fullMapImage is None:
            print("Need to interpolate")
            self.__fullMapImage = environment.hsv_gradient()
        
        self.__xEnv = self.__environment.cloud[0, :, 0]
        self.__yEnv = self.__environment.cloud[:, 0, 1]
    
        self.__xEnv = [i for i in np.arange(self.__xEnv[0], self.__xEnv[-1] + precision, precision)]
        self.__yEnv = [i for i in np.arange(self.__yEnv[0], self.__yEnv[-1] + precision, precision)]
    
    def compute_localization(self, i, previousState, insMeasures, previousInsMeasure, view):
        
        C, y, Gamma_obs = self.__generate_observations(i, view)
        
        
        
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
    
        return estimatedState
        
    
    def __generate_observations(self, i, view):
        
        C = None
        y = None
        Gamma_obs = None
        
        if len(view):
        
            view = view[:,:,2]
    
            view_img = cv2.imread("Environment_data/area_1/Images/RGB/RGB_sonar_view_state{}.png".format(i))
        
            if view_img is None:
                print("Load failed")
                view_img = self.__converter.img_gradient(view.T, self.__noise)
            
            start = time.time()
            _, R, T = self.__matcher.matchImages(view_img, self.__fullMapImage, False)
            TBN_SURF_LocalizationEstimator.match_time += time.time() - start
            
            
            if not((R is None) or (T is None)):
                
                center = np.array([[50],[50]])
                center = R@center + T
                
                if 0 <= center[0] <= len(self.__xEnv) and 0 <= center[1] <= len(self.__yEnv):
                
                    X = self.__xEnv[int(center[0])] + (center[0] - int(center[0]))*self.__precision
                    Y = self.__yEnv[int(center[1])] + (center[1] - int(center[1]))*self.__precision
                    
                    y = np.array([X[0], Y[0]])
                    
                    C = np.array([[1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0]])
    
                    
                    Gamma_obs = 10*np.eye(2)
        
        return C, y, Gamma_obs
    
if __name__ == '__main__':
    
    environment = Environment("cloud.npz")
    
    estimator = TBN_SURF_LocalizationEstimator(environment)
    
    estimator.compute_localization(0, 0)