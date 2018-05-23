import numpy as np


class RobotState:
    """Describe the current state of the robot and store the previous states"""

    def __init__(self, pose = np.array([0, 0, 0]), orientation = np.array([0, 0, 0]), linear_velocity = np.array([0, 0, 0]), angular_velocity = np.array([0, 0, 0])):
        self.pose 		= pose
        self.orientation 	= orientation
        self.linear_velocity 	= linear_velocity
        self.angular_velocity 	= angular_velocity

    def __str__(self):
        return "Position    : {}\nOrientation : {}\nVelocity    : {}\n".format(self.pose, self.orientation, self.linear_velocity)

    def updateState(self, dt, u):
        Vx = self.linear_velocity[0]
        yaw = self.orientation[2]
        
        new_pose = self.pose + dt*Vx*np.array([np.cos(yaw), \
                                               np.sin(yaw), \
                                               0])
                                               
        new_orientation = self.orientation + dt*np.array([0.0, \
                                                         0.0, \
                                                         u])
                                                         
        new_linear_velocity = self.linear_velocity
        
        new_angular_velocity = self.angular_velocity
        
        return RobotState(new_pose, new_orientation, new_linear_velocity, new_angular_velocity)

    # def plot(self, axis):
    #     draw_auv3D(axis, self.pose, self.orientation[0], self.orientation[1], self.orientation[2], 'blue') 
