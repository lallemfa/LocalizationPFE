from math import sqrt
import matplotlib.pyplot as plt

class Simulation:

    def __init__(self, dt, environment, missions, robot):
        self.dt             = max(1e-5, dt)
        self.environment    = environment
        self.missions       = missions
        self.robot          = robot

    def __str__(self):
        return "Simulation\n\tdt -> {}\n".format(self.dt)

    def validateWaypoint(self, waypoint):
        robot_pose = self.robot.true_curr_state.pose
        
        return sqrt(sum((waypoint - robot_pose)**2)) < 2.0
        
    def run(self, plot_flag = 0):
        for mission in self.missions:
            self.runMission(mission, plot_flag)
        
        return 0
    
    def runMission(self, mission, plot_flag = 0):
        
        fig 	= plt.figure("Run plot")
        ax 	= fig.add_subplot(111, projection = '3d')
    
        print("Begin run mission")
        
        for i in range(len(mission.waypoints[0])):
            waypoint = mission.waypoints[:, i]
            while not self.validateWaypoint(waypoint):
                self.robot.updateState(self.dt, waypoint)
                
                if plot_flag:
                    ax.clear()
                    
                    mission.plot(ax)
                    self.robot.plot(ax)
                    
                    plt.show()
                    
                    plt.pause(1e-15)
                    
        print("End run mission")
        
        self.plot(ax, mission)
        
    def plot(self, axis, mission):
        axis.clear()
        
        mission.plot(axis)
        self.environment.plot(axis, 15)
        self.robot.plot(axis)
        
        plt.legend()
        
        plt.show()