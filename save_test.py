# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 11:05:47 2018

@author: fabrice.lallement
"""

from Environment import Environment
from Mission import Mission
from Robot import Robot
from Simulation import Simulation
from INS import INS
from Sonar import Sonar

import dill

environment = Environment("cloud.npz")
mission = Mission(environment)

ins = INS(linear_bias = [0, 0, 0], \
         linear_white_noise_standard_deviation = [0, 0, 0], \
         linear_bias_instability_standard_deviation = [0, 0, 0], \
         angular_bias = [0, 0, 0], \
         angular_white_noise_standard_deviation = [0, 0, 0], \
         angular_bias_instability_standard_deviation = [0, 0, 0])

sonar = Sonar(environment, [200, 200], [100, 100])

robot = Robot([4.49652985e+5, 4.951035090e+06, 0], ins = ins, sonar = sonar)

simulation = Simulation(0.5, environment, [mission], robot)

filename = 'globalsave.pkl'
dill.dump_session(filename)