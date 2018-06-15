from RobotState import RobotState
from StructMatch import StructMatch
from INS import INS
import numpy as np
from roblib import sawtooth, rotmat, kalman, draw_ellipse

import matplotlib.pyplot as plt

import cv2


class Robot:
    """Describe the current state of the robot and store the previous states"""

    def __init__(self, pose = np.array([0, 0, 0]), \
                 orientation = np.array([0, 0, 0]), \
                 linear_velocity = np.array([1, 0, 0]), \
                 angular_velocity = np.array([0, 0, 0]), \
                 initial_gamma = np.zeros((3, 3)), \
                 ins = None, \
                 sonar = None):
        
        self.true_curr_state = self.estimated_curr_state = RobotState(pose, orientation, linear_velocity, angular_velocity, initial_gamma)
        
        self.true_state_history         = [self.true_curr_state]
        self.estimated_state_history    = []
        
        self.histories = [self.true_state_history]
        
        self.labels = ["Ground truth", "Dead reckoning"]
        
        self.ins = ins
        if not ins is None:
            self.gamma_ins = self.ins.getCovarianceOfMeasures()
            
        self.sonar = sonar

        self.previous_measure = np.array([0, 0, -9.81, 0, 0, 0])
        
        self.i = 0
        
        self.views = None
        
# =============================================================================
#         Equations
# =============================================================================
        
        self.A = np.eye(3)
        self.B = np.eye(3)
        
# =============================================================================
#         StructMatcher
# =============================================================================
        
        self.matcher = StructMatch()
        self.full_map_img = cv2.imread('full_map_hsv.png')

    def __str__(self):
        return "Number of state(s) in trajectory : {}\n".format(len(self.true_state_history))
        
    def control(self, waypoint):
        curr_pos = self.true_curr_state.pose
        wp_gisement = np.arctan2(waypoint[1] - curr_pos[1], waypoint[0] - curr_pos[0])
        
        yaw_error = sawtooth(self.true_curr_state.orientation[2] - wp_gisement)
        
        if abs(yaw_error) > 2.5:
            u = 1.0
        else:
            u = -0.1 * yaw_error
        
        return u
    
    def updateState(self, dt, waypoint):
        
        u  = self.control(waypoint)
        
        pose, angles, velocities, angular_velocity = self.true_curr_state.updateState(dt, u)
        
        if not (self.sonar is None) and self.i%50 == 0:
            _, view = self.sonar.generateView(pose[0:2].T, angles[2])
        else:
            view = []
            
        self.i += 1
        
        next_state = RobotState(pose, angles, velocities, angular_velocity, view = view)   
        
        if (len(self.estimated_state_history) == 0):
            self.estimated_state_history    = [self.true_curr_state, next_state]
            
            self.histories.append(self.estimated_state_history)
        
        if (len(self.true_state_history) >= 2) and not (self.ins is None):
            state_minus_2 = self.true_state_history[-2]
        
            measures = self.ins.generateMeasures(dt, next_state, self.true_curr_state, state_minus_2)
            
            pose, angles, velocities, angular_velocity, Gamma = self.integrateINS(dt, measures, self.previous_measure, self.estimated_state_history)
            
            self.previous_measure = measures
        
            # =============================================================================
            #         Create state instance
            # =============================================================================
                    
            self.estimated_curr_state = RobotState(pose, angles, velocities, angular_velocity, Gamma, view = view)   
            
            # =============================================================================
            #         Save state
            # =============================================================================
                    
            self.estimated_state_history.append(self.estimated_curr_state)
        

        self.true_curr_state = next_state
        
        self.true_state_history.append(self.true_curr_state)
        
        
        
        
    def integrateINS(self, dt, measures, previous_measure, history):
        
        correct_gravity = np.array([0, 0, 9.81])
        
        previous_state = history[-1]
        
# =============================================================================
#         Angular integration
# =============================================================================
        
        angular_velocity = measures[3:6]
        
        angles = dt*angular_velocity + previous_state.orientation
        
        R = rotmat(angles[0], angles[1], angles[2])
        
# =============================================================================
#         Velocities integration after correction of rotation and gravity
# =============================================================================
        
        measures[0:3] = R@measures[0:3]
        
        velocities = dt*(previous_measure[0:3] + measures[0:3] + 2*correct_gravity)/2 + previous_state.linear_velocity
        
# =============================================================================
#         Kalman filter for computing pose
# =============================================================================
        
        pose, Gamma = kalman(previous_state.pose, self.A, previous_state.gamma, dt*velocities, self.B, self.gamma_ins)
        
        return pose, angles, velocities, angular_velocity, Gamma
        
    def plot(self, axis):
        
        for history, label in zip(self.histories, self.labels):
                    
            if len(history) >= 2:
                trajectory = []
                for state in history:
                    trajectory = np.vstack((trajectory, state.pose)) if len(trajectory) else state.pose
        
                axis.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label = label)
    
    def plot_uncertainty(self):
        
        fig = plt.figure("Uncertainty XY-plane")
        
        axis = fig.add_subplot(111, aspect='equal')
        
        for history, label in zip(self.histories, self.labels):
            if len(history) >= 2:
                trajectory = []
                i = 0
                for state in history:
                    if i%40 == 0:
                        draw_ellipse(state.pose[0:2], state.gamma[0:2, 0:2], 0.9, axis, 'r')
                        
                    i += 1
                    
                    trajectory = np.vstack((trajectory, state.pose[0:2])) if len(trajectory) else state.pose[0:2]
                
                axis.plot(trajectory[:, 0], trajectory[:, 1], '-.', label = label)            
        
        plt.show()

    def plot_history(self):
        
        if len(self.true_state_history) >= 2:
            true_history = []
            for state in self.true_state_history:
                
                data = np.hstack((state.pose, state.orientation, state.linear_velocity))
                
                true_history = np.vstack((true_history, data)) if len(true_history) else data
        
        if len(self.estimated_state_history) >= 2:
            estimated_history = []
            for state in self.estimated_state_history:
                
                data = np.hstack((state.pose, state.orientation, state.linear_velocity))
                
                estimated_history = np.vstack((estimated_history, data)) if len(estimated_history) else data
        
        
# =============================================================================
#         Position plot
# =============================================================================
        
        f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        
        ax1.plot(true_history[:, 0], label = "True X")
        ax1.plot(estimated_history[:, 0], '-.', label = "Estimated X")
        ax1.legend()
        
        ax2.plot(true_history[:, 1], label = "True Y")
        ax2.plot(estimated_history[:, 1], '-.', label = "Estimated Y")
        ax2.legend()
        
        ax3.plot(true_history[:, 2], label = "True Z")
        ax3.plot(estimated_history[:, 2], '-.', label = "Estimated Z")
        ax3.legend()
        
        plt.show()
        
# =============================================================================
#         Angles plot
# =============================================================================
        
        f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        
        ax1.plot(true_history[:, 3], label = "True roll")
        ax1.plot(estimated_history[:, 3], '-.', label = "Estimated roll")
        ax1.legend()
        
        ax2.plot(true_history[:, 4], label = "True pitch")
        ax2.plot(estimated_history[:, 4], '-.', label = "Estimated pitch")
        ax2.legend()
        
        ax3.plot(true_history[:, 5], label = "True yaw")
        ax3.plot(estimated_history[:, 5], '-.', label = "Estimated yaw")
        ax3.legend()
        
        plt.show()
        
# =============================================================================
#         Velocities plot
# =============================================================================
        
        f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        
        ax1.plot(true_history[:, 6], label = "True X axis")
        ax1.plot(estimated_history[:, 6], '-.', label = "Estimated X axis")
        ax1.legend()
        
        ax2.plot(true_history[:, 7], label = "True Y axis")
        ax2.plot(estimated_history[:, 7], '-.', label = "Estimated Y axis")
        ax2.legend()
        
        ax3.plot(true_history[:, 8], label = "True Z axis")
        ax3.plot(estimated_history[:, 8], '-.', label = "Estimated Z axis")
        ax3.legend()
        
        plt.show()
        
        
        return
    
    def extractViewsCentersGamma(self):
    
        views = []
        centers = []
        gamma = []
        
        if len(self.estimated_state_history) >= 2:
            
            for state in self.estimated_state_history:
                
                if len(state.view):
                    psi = state.orientation[2]
                    
                    R = np.array([[np.cos(psi), -np.sin(psi), 0],
                                  [np.sin(psi),  np.cos(psi), 0],
                                  [          0,            0, 1]])
                    
                    view = np.reshape(state.view, (-1, 1, 3))[:, 0, :]
                    
                    view = R@view.T
                    
                    view[0, :] += state.pose[0]
                    view[1, :] += state.pose[1]
                    
                    views = np.dstack((views, state.view[:,:,2])) if len(views) else state.view[:,:,2]
                    
                    centers = np.dstack((centers, state.pose[0:2])) if len(centers) else state.pose[0:2]
                    
                    gamma = np.dstack((gamma, state.gamma[0:2, 0:2])) if len(gamma) else state.gamma[0:2, 0:2]

        return views, centers, gamma
    
    def computeTrajectoryWithNewINS(self, ins, dt, name = "No name"):
        
        new_estimated_history = []
        previous_measure = np.array([0, 0, -9.81, 0, 0, 0])
        
        if len(self.true_state_history) >= 2:
            
            new_estimated_history = self.true_state_history[0:2]
            
            for i in range(2, len(self.true_state_history)):
                
                current_state = self.true_state_history[i]
                minus_1_state = self.true_state_history[i-1]
                minus_2_state = self.true_state_history[i-2]
                
                view = self.true_state_history[i].view
                
                measures = ins.generateMeasures(dt, current_state, minus_1_state, minus_2_state)
                
                pose, angles, velocities, angular_velocity, Gamma = self.integrateINS(dt, measures, previous_measure, new_estimated_history)
                
                previous_measure = measures
                
                # =============================================================================
                #         Create state instance
                # =============================================================================
                        
                newly_estimated_curr_state = RobotState(pose, angles, velocities, angular_velocity, Gamma, view = view)   
                
                # =============================================================================
                #         Save state
                # =============================================================================
                        
                new_estimated_history.append(newly_estimated_curr_state)
        
        self.histories.append(new_estimated_history)
        self.labels.append(name)
        
        return new_estimated_history
    
    def generateObservationForKalman(self, data, method):
        
        C = 0
        y = 0
        Gamma_obs =0
        
        if len(data):
            if method == "TBN_SURF":
                C, y, Gamma_obs = self.generateObservationForTBN(data)
            elif method == "ScanMatching":
                print("Not implemented yet")
                pass
            
        return C, y, Gamma_obs
    
    def generateObservationForTBN(self, data):
        
        C = 0
        y = 0
        Gamma_obs =0
        
        center, R, T = self.matcher.matchImages(data, self.full_map_img, False)
        
        return C, y, Gamma_obs
        
if __name__ == '__main__':

    robot = Robot(np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0]))











