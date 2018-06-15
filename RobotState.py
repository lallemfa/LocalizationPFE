import numpy as np


class RobotState:
    """Describe the current state of the robot and store the previous states"""

    def __init__(self, pose = np.array([0, 0, 0]), \
                 orientation = np.array([0, 0, 0]), \
                 linear_velocity = np.array([0, 0, 0]), \
                 angular_velocity = np.array([0, 0, 0]), \
                 gamma_state = None, \
                 view = []):
        
        self.pose 		        = np.asarray(pose)
        self.orientation        = np.asarray(orientation)
        self.linear_velocity    = np.asarray(linear_velocity)
        self.angular_velocity 	 = np.asarray(angular_velocity)
        
        
        self.gamma = gamma_state if not gamma_state is None else np.zeros( ( len(self.pose) + len(self.orientation),
                                                                             len(self.pose) + len(self.orientation) ) )
        
        self.gamma = gamma_state if not gamma_state is None else np.zeros( ( len(self.pose),
                                                                             len(self.pose) ) )
        
        self.view = view

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
        
        return new_pose, new_orientation, new_linear_velocity, new_angular_velocity

    # def plot(self, axis):
    #     draw_auv3D(axis, self.pose, self.orientation[0], self.orientation[1], self.orientation[2], 'blue') 
