import numpy as np
import matplotlib.pyplot as plt
import time

from Environment import Environment
from Mission import Mission
from Robot import Robot
from Simulation import Simulation



start = time.time()

environment = Environment("cloud.npz")
mission = Mission(environment)
robot = Robot(np.array([4.49652985e+5, 4.951035090e+06, 0]))
simulation = Simulation(0.5, environment, [mission], robot)

simulation.run(0)

print( "Time elapsed : {} seconds.".format(time.time() - start) )
