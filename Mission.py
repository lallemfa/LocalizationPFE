import numpy as np
import time

class Mission:

    def __init__(self, environment, waypoints = []):
       
        self.environment = environment
        self.waypoints = waypoints if len(waypoints) else self.generateWaypoints()
	    
    def __str__(self):
        return "Number of waypoints : {}\n\nWaypoints :\n{}".format(len(self.waypoints), self.waypoints)


    def generateWaypoints(self, number_of_lines = 10):
	
        waypoints = []
        	
        limits = self.environment.limitsOfEnvironment()
        	    
        gap_x = limits[1] - limits[0]
        gap_y = limits[3] - limits[2]
        	    
        coord_y = [ limits[2] + 0.15*gap_y , limits[3] - 0.15*gap_y ]
        	    
        percent = [i for i in np.linspace(0, 1, number_of_lines)]
	    
        for i in percent:
            val_x = limits[0] + (0.15 + i*0.70)*gap_y
            coord_x = [ val_x, val_x ]
            
            coord_z = np.zeros(shape=(1, 2))
	        
            line_waypoints = np.vstack((coord_x, coord_y, coord_z))
            	        
            coord_y.reverse()
            	        	        
            waypoints = np.hstack((waypoints, line_waypoints)) if len(waypoints) else line_waypoints
	    
        return waypoints

    def plot(self, axis):
        plot = axis.plot(self.waypoints[0, :], self.waypoints[1, :], 'bd', label='Waypoints')
        return plot

