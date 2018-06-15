import time

#import matplotlib.pyplot as plt
#import numpy as np

from Environment import Environment
from Mission import Mission
from Robot import Robot
from Simulation import Simulation
from INS import INS
from Sonar import Sonar

import dill

#import cv2

start = time.time()

environment = Environment("cloud.npz")
mission = Mission(environment)

ins = INS(linear_bias = [0, 0, 0], \
         linear_white_noise_standard_deviation = [0, 0, 0], \
         linear_bias_instability_standard_deviation = [0, 0, 0], \
         angular_bias = [0, 0, 0], \
         angular_white_noise_standard_deviation = [0, 0, 0], \
         angular_bias_instability_standard_deviation = [0, 0, 0])

#sonar = Sonar(environment, [100, 100])

sonar = Sonar(environment, [200, 200], [100, 100])

robot = Robot([4.49652985e+5, 4.951035090e+06, 0], ins = ins, sonar = sonar)

simulation = Simulation(0.5, environment, [mission], robot)

simulation.run()

print( "Time elapsed : {} seconds.".format(time.time() - start) )

filename = 'run_data_cubic_200x200_100x100_without_noise.pkl'
dill.dump_session(filename)

#robot.plot_history()

#robot.plot_uncertainty()

#ins2 = INS(linear_bias = [0, 0, 0], \
#         linear_white_noise_standard_deviation = [1e-4, 0, 0], \
#         linear_bias_instability_standard_deviation = [0, 1e-8, 0], \
#         angular_bias = [0, 0, 0], \
#         angular_white_noise_standard_deviation = [0, 0, 0], \
#         angular_bias_instability_standard_deviation = [0, 0, 0])
#
#robot.computeTrajectoryWithNewINS(ins2, 0.5, "INS 2")
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

#views, centers, gamma = robot.extractViewsCentersGamma()
#
#
#
#[Gx, Gy] = environment.gradient()
#ori_hsv_img = environment.hsv_gradient(Gx, Gy, environment.cloud[:, :, 2])
#
#cv2.imshow("Map", cv2.cvtColor(ori_hsv_img, cv2.COLOR_HSV2BGR))
#
#for i in range(views.shape[2]):
#    
#    view    = views[:, :, i]
#    center  = centers[0, :, i]
#    
#    idx_x = np.where(np.sort(np.hstack((environment.cloud[0, :, 0], center[0]))) == center[0])
#    idx_y = np.where(np.sort(np.hstack((environment.cloud[:, 0, 1], center[1]))) == center[1])
#    
#    min_x = max(0, idx_x[0][0] - 150)
#    max_x = min(500, idx_x[0][0] + 150)
#    
#    min_y = max(0, idx_y[0][0] - 150)
#    max_y = min(500, idx_y[0][0] + 150)
#    
#    matcher = Matcher(environment, "SURF")
#    
#    [Gx, Gy] = environment.gradient(view.T)
#    
#    sonar_hsv_img = environment.hsv_gradient(Gx, Gy, view.T)
#    
#    known_hsv_img = ori_hsv_img[min_y:max_y, min_x:max_x, :]
#    
#    cv2.imwrite("Images/HQ_sonar_img_{}.png".format(i), sonar_hsv_img)
#    
#    cv2.imwrite("Images/HQ_map_img_{}.png".format(i), known_hsv_img)
    
#    matches, _, _, _, _ = matcher.match_imgs(known_hsv_img, sonar_hsv_img)
#    
#    if not matches is None:
#        cv2.waitKey(0)

    
    