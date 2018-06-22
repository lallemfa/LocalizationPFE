from LocalizationEstimator import LocalizationEstimator
from roblib import rotmat, kalman
from RobotState import RobotState

import numpy as np

class Dead_Reckoning_LocalizationEstimator(LocalizationEstimator):
    
    def __init__(self, dt, A, B, gammaINS):
        
        super(Dead_Reckoning_LocalizationEstimator, self).__init__(dt)
        
        self.A = A
        self.B = B
        
        self.gammaINS = gammaINS
    
    def compute_localization(self, i, previousState, insMeasures, previousInsMeasure, view):
        
        C, y, Gamma_obs = self.__generate_observations()
        
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
        
    
    def __generate_observations(self):
        
        C = None
        y = None
        Gamma_obs = None
        
        return C, y, Gamma_obs
        
    
if __name__ == '__main__':
    
    estimator = Dead_Reckoning_LocalizationEstimator()
    
    estimator.compute_localization(0, 0, 0, 0)
    
    
    
    
    