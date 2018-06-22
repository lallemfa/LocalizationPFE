# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 15:21:03 2018

@author: fabrice.lallement
"""

import time
import dill
import numpy as np
import cv2

from Environment import Environment
from Mission import Mission
from Robot import Robot
from Simulation import Simulation
from Sonar import Sonar
from DepthmapToImage import DepthmapToImage

# =============================================================================
# Load data
# =============================================================================

filename = "Environment_data/area_3/cubic_200x200_100x100_without_ins.pkl"

dill.load_session(filename)

converter = DepthmapToImage()

img = converter.init_with_environment(environment)

#cv2.imwrite("Environment_data/area_3/Images/RGB/full_map_rgb_2m.png", img)

for i in range(1, 11):
    
    img = converter.init_with_environment(environment, i)
    cv2.imwrite("Environment_data/area_3/Images/RGB/new_map_rgb_{}m.png".format(i), img)
#
#for i in range(1, 11):
#    
#    cloud = environment.reduceResolution(i)
#    
#    img = converter.img_gradient(cloud[:, :, 2])
#    cv2.imwrite("Environment_data/area_3/Images/RGB/0_map_rgb_{}m.png".format(i), img)

#img = converter.init_with_environment(environment)

#history = robot.histories[0]
#
#for i in range(len(history)):
#    state = history[i]
#    view = state.view
#    
#    if len(view):
#        view = view[:, :, 2]
#        hsv_img = converter.img_gradient(view.T)
#        
#        cv2.imwrite("Environment_data/area_3/Images/RGB/RGB_sonar_view_state{}.png".format(i), hsv_img)