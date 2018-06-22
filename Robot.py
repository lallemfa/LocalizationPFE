from RobotState import RobotState
import numpy as np
from roblib import sawtooth, draw_ellipse

import matplotlib.pyplot as plt
import time

from Dead_Reckoning_LocalizationEstimator import Dead_Reckoning_LocalizationEstimator

class Robot:
    """Describe the current state of the robot and store the previous states"""

    def __init__(self, pose = np.array([0, 0, 0]), \
                 orientation = np.array([0, 0, 0]), \
                 linear_velocity = np.array([1, 0, 0]), \
                 angular_velocity = np.array([0, 0, 0]), \
                 initial_gamma = np.zeros((3, 3)), \
                 ins = None, \
                 sonar = None, \
                 environment = None):
        
        self.true_curr_state = RobotState(pose, orientation, linear_velocity, angular_velocity, initial_gamma)
        
        self.true_state_history = [self.true_curr_state]
        
        
        self.ins_measures_history = []
        
        
        self.histories = [self.true_state_history]
        
        self.labels = ["Ground truth"]
        
# =============================================================================
#         Equations
# =============================================================================
        
        self.A = np.eye(3)
        self.B = np.eye(3)
        
        self.ins = ins
        if not ins is None:
            self.estimated_state_history    = []
            self.labels.append("Dead reckoning")
            self.gamma_ins = self.ins.getCovarianceOfMeasures()
            
            self.deadReckoningEstimator = Dead_Reckoning_LocalizationEstimator(0.5, np.eye(3), np.eye(3), self.gamma_ins)
            
        self.sonar = sonar

        self.previous_measure = np.array([0, 0, -9.81, 0, 0, 0])
        
        self.i = 0
        
        
        
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
        
        self.true_curr_state = RobotState(pose, angles, velocities, angular_velocity, view = view)   
        
        self.true_state_history.append(self.true_curr_state)
        
# =============================================================================
#         Live INS if possible
# =============================================================================
        
        if not(self.ins is None) and len(self.true_state_history) >= 3:
            
            state_minus_1 = self.true_state_history[-2]
            state_minus_2 = self.true_state_history[-3]
            
            if (len(self.estimated_state_history) == 0):
                self.estimated_state_history.append(self.true_state_history[0])
                self.estimated_state_history.append(self.true_state_history[1])
                
                self.histories.append(self.estimated_state_history)
                self.labels.append("Dead reckoning")
            
            measures = self.ins.generateMeasures(dt, self.true_curr_state, state_minus_1, state_minus_2)
            
            self.ins_measures_history = np.vstack((self.ins_measures_history, measures)) \
                                        if len(self.ins_measures_history) else measures
            
            previousState = self.estimated_state_history[-1]
            
            estimatedState = self.deadReckoningEstimator.compute_localization(previousState, measures, self.previous_measure, view)
            
            self.previous_measure = measures
            
            self.estimated_state_history.append(estimatedState)
    
    def replace_INS_measures(self, dt, ins, name = "INS"):
        
        ins_history = self.conpute_ins_measure_history(dt, ins)
        
        self.ins_measures_history = np.copy(ins_history)
        
        deadReckoningEstimator = Dead_Reckoning_LocalizationEstimator(dt, np.eye(3), np.eye(3), ins.getCovarianceOfMeasures())
        
        self.compute_new_trajectory(deadReckoningEstimator, name = name)
        
        INS_trajectory_history = self.histories.pop()
        self.histories.insert(1, INS_trajectory_history)
        
        self.estimated_state_history = INS_trajectory_history
        
        
    def conpute_ins_measure_history(self, dt, ins):
        
        ins_history = None
        
        for i in range(2, len(self.true_state_history)):
            current_state = self.true_state_history[i]
            minus_1_state = self.true_state_history[i-1]
            minus_2_state = self.true_state_history[i-2]

            measures = ins.generateMeasures(dt, current_state, minus_1_state, minus_2_state)
        
            ins_history = np.vstack((ins_history, measures)) if (not ins_history is None) else measures
        
        return ins_history
    
    def compute_new_trajectory(self, estimator, ins = None, name = "No name"):
        
        dt = estimator.dt
        
        estimated_history = []
        previous_measure = np.array([0, 0, -9.81, 0, 0, 0])
        
        if len(self.true_state_history) >= 2:
            
            estimated_history = self.true_state_history[0:2]
            
            if ins is None:
                ins_measures_history = self.ins_measures_history
            else:
                ins_measures_history = self.conpute_ins_measure_history(dt, ins)
                
                
            for i in range(2, len(self.true_state_history)):
                
                data = self.true_state_history[i].view
                
                if name == "ScanMatching" or name == "Noisy ScanMatching":
                    
                    previousView = self.true_state_history[i-50].view
                    previousPose = self.true_state_history[i-50].pose
                    previousAngle = self.true_state_history[i-50].orientation
                    view = self.true_state_history[i].view
                    
                    data = [previousView, view, previousPose[0:2], previousAngle[2]]
                
                measures = ins_measures_history[i-2, :]
                
                previousState = estimated_history[-1]
                
                estimated_state = estimator.compute_localization(i, previousState, measures, previous_measure, data)
                
                previous_measure = measures
                
                estimated_history.append(estimated_state)
                
                
        
        self.histories.append(estimated_history)
        self.labels.append(name)
        
        return estimated_history
    
# =============================================================================
# =============================================================================
# #     PLOT FUNCTIONS
# =============================================================================
# =============================================================================
    
    def plot(self, axis = None):
        
        if axis is None:
            fig = plt.figure("Robot trajectories")
            axis = fig.add_subplot(111, aspect="equal")
        
        for history, label in zip(self.histories, self.labels):
                    
            if len(history) >= 2:
                trajectory = []
                for state in history:
                    trajectory = np.vstack((trajectory, state.pose)) if len(trajectory) else state.pose
        
                axis.plot(trajectory[:, 0], trajectory[:, 1], label = label)
        
        axis.legend()
        
    def plot_uncertainty(self):
        
#        fig = plt.figure("Uncertainty XY-plane")
#        traj_axis = fig.add_subplot(111, aspect='equal')
        
        fig = plt.figure("Evolution of error")
        error_axis = fig.add_subplot(111)
        
        init = True
        
        for history, label in zip(self.histories, self.labels):
            if len(history) >= 2:
                trajectory = []
                i = 0
                for state in history:
#                    if i%200 == 0:
#                        draw_ellipse(state.pose[0:2], state.gamma[0:2, 0:2], 0.9, traj_axis, 'r')
                        
                    i += 1
                    
                    trajectory = np.vstack((trajectory, state.pose[0:2])) if len(trajectory) else state.pose[0:2]
                
#                traj_axis.plot(trajectory[:, 0], trajectory[:, 1], '-.', label = label)
                
                if init:
                    ground_truth = np.copy(trajectory)
                    init = False
                else:
                    error = np.sqrt((trajectory[:, 0] - ground_truth[:, 0])**2 +
                                    (trajectory[:, 1] - ground_truth[:, 1])**2)
                    
                    error_axis.plot(error, '-.', label = label)
        
#        traj_axis.legend()
        error_axis.legend()
        plt.show()
    
    def plot_history(self):
        
        _, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.set_title('X Axis')
        ax2.set_title('Y Axis')
        ax3.set_title('Z Axis')
        
        _, (ax4, ax5, ax6) = plt.subplots(1, 3)
        ax4.set_title('Roll')
        ax5.set_title('Pitch')
        ax6.set_title('Yaw')
        
        for history, label in zip(self.histories, self.labels):
            if len(history) >= 2:
                trajectory = []
                for state in history:
                    data = np.hstack((state.pose, state.orientation, state.linear_velocity))
                    trajectory = np.vstack((trajectory, data)) if len(trajectory) else data
                
                ax1.plot(trajectory[:, 1])
                ax2.plot(trajectory[:, 2])
                ax3.plot(trajectory[:, 3])
                ax4.plot(trajectory[:, 4])
                ax5.plot(trajectory[:, 5])
                ax6.plot(trajectory[:, 6])
  
        plt.show()

    def comparison_plot(self):
        
        if len(self.histories) >= 2:
            
# =============================================================================
#             Ground truth
# =============================================================================
            
            groundTruth = self.histories[0]
            groundTruthLabel = self.labels[0]
            
            if len(groundTruth) >= 2:
                groundTruthTrajectory = []
                for state in groundTruth:
                    
                    groundTruthTrajectory = np.vstack((groundTruthTrajectory, state.pose[0:2])) if len(groundTruthTrajectory) else state.pose[0:2]
            
# =============================================================================
#             Dead Reckoning
# =============================================================================
            
            deadReckoning = self.histories[1]
            deadReckoningLabel = self.labels[1]
            
            if len(deadReckoning) >= 2:
                deadReckoningTrajectory = []
                for state in deadReckoning:
                    
                    deadReckoningTrajectory = np.vstack((deadReckoningTrajectory, state.pose[0:2])) if len(deadReckoningTrajectory) else state.pose[0:2]
       
                deadReckoningError = np.sqrt((deadReckoningTrajectory[:, 0] - groundTruthTrajectory[:, 0])**2 +
                                             (deadReckoningTrajectory[:, 1] - groundTruthTrajectory[:, 1])**2)
                
            
                plt.figure("{} | {}".format(groundTruthLabel, deadReckoningLabel))
                ax1 = plt.subplot2grid((1, 3), (0, 0), aspect='equal')
                ax2 = plt.subplot2grid((1, 3), (0, 1), colspan=2)
                
                ax1.plot(groundTruthTrajectory[:, 0], groundTruthTrajectory[:, 1], 'g', label = groundTruthLabel)
                ax1.plot(deadReckoningTrajectory[:, 0], deadReckoningTrajectory[:, 1], 'b', label = deadReckoningLabel)
                ax1.legend()
                
                ax2.plot(deadReckoningError, label = deadReckoningLabel)
                ax2.legend()
        
# =============================================================================
#         Others
# =============================================================================
        
        if len(self.histories) >= 3:
            
            for history, label in zip(self.histories[2:], self.labels[2:]):
                
                plt.figure("{}".format(label))
                ax1 = plt.subplot2grid((1, 3), (0, 0), aspect='equal')
                ax2 = plt.subplot2grid((1, 3), (0, 1), colspan=2)
                
                ax1.plot(groundTruthTrajectory[:, 0], groundTruthTrajectory[:, 1], 'g', label = groundTruthLabel)
                ax1.plot(deadReckoningTrajectory[:, 0], deadReckoningTrajectory[:, 1], 'b', label = deadReckoningLabel)
            
                ax2.plot(deadReckoningError, label = deadReckoningLabel)
            
                if len(history) >= 2:
                    trajectory = []
                    for state in history:
                        trajectory = np.vstack((trajectory, state.pose[0:2])) if len(trajectory) else state.pose[0:2]
                    
                    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'r', label = label)
            
                    error = np.sqrt((trajectory[:, 0] - groundTruthTrajectory[:, 0])**2 +
                                    (trajectory[:, 1] - groundTruthTrajectory[:, 1])**2)
                    
                    ax2.plot(error, label = label)
            
                ax1.legend()
                ax2.legend()

        plt.show()
        
if __name__ == '__main__':

    robot = Robot(np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0]))








