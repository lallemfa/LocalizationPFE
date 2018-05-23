from RobotState import RobotState
import numpy as np
from math import pi

def sawtooth(x):
    return (x+pi)%(2*pi)-pi

class Robot:
    """Describe the current state of the robot and store the previous states"""

    def __init__(self, pose = np.array([0, 0, 0]), orientation = np.array([0, 0, 0]), linear_velocity = np.array([1, 0, 0]), angular_velocity = np.array([0, 0, 0])):
        self.curr_state     = RobotState(pose, orientation, linear_velocity, angular_velocity)
        self.state_history  = [self.curr_state]

    def __str__(self):
        return "Number of state(s) in trajectory : {}\n".format(len(self.state_history))
        
    def control(self, waypoint):
        curr_pos = self.curr_state.pose
        wp_gisement = np.arctan2(waypoint[1] - curr_pos[1], waypoint[0] - curr_pos[0])
        
        yaw_error = sawtooth(self.curr_state.orientation[2] - wp_gisement)
        
        if abs(yaw_error) > 2.5:
            u = 1.0
        else:
            u = -0.1 * yaw_error
        
        return u
    
    def updateState(self, dt, waypoint):
        u  = self.control(waypoint)
        self.curr_state = self.curr_state.updateState(dt, u)
        self.state_history.append(self.curr_state)

    def plot(self, axis):

        trajectory = []

        for state in self.state_history:
            trajectory = np.vstack((trajectory, state.pose)) if len(trajectory) else state.pose

        draw = axis.plot(trajectory[:, 0], trajectory[:, 1])

        return draw

if __name__ == '__main__':

    robot = Robot(np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0]))
