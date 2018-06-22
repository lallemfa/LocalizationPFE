import time

#import matplotlib.pyplot as plt
#import numpy as np

from Environment import Environment
from Mission import Mission
from Robot import Robot
from Simulation import Simulation
#from INS import INS
from Sonar import Sonar

import dill

start = time.time()

environment = Environment("Environment_data/area_4/map.npz")
mission = Mission(environment)

x_start = environment.cloud[0, 0, 0]
y_start = environment.cloud[0, 0, 1]

sonar = Sonar(environment, [200, 200], [100, 100])

robot = Robot([x_start, y_start, 0], sonar = sonar)


simulation = Simulation(0.5, environment, [mission], robot)

simulation.run()

print( "Time elapsed : {} seconds.".format(time.time() - start) )

filename = "Environment_data/area_4/cubic_200x200_100x100_without_ins.pkl"
dill.dump_session(filename)

#robot.plot_history()

#robot.plot_uncertainty()
#
#ins = INS(linear_bias = [0, 0, 0], \
#         linear_white_noise_standard_deviation = [1e-4, 0, 0], \
#         linear_bias_instability_standard_deviation = [0, 1e-8, 0], \
#         angular_bias = [0, 0, 0], \
#         angular_white_noise_standard_deviation = [0, 0, 0], \
#         angular_bias_instability_standard_deviation = [0, 0, 0])
#
#robot.computeTrajectoryWithNewINS(ins, 0.5, "INS ")
#
#robot.plot_uncertainty()
#
#fig = plt.figure("Run plot 2")
#ax 	= fig.add_subplot(111, projection = '3d')
#
#mission.plot(ax)
#robot.plot(ax)
#
#plt.legend()
#
#plt.show()

    
    