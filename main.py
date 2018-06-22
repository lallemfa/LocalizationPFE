import time
import dill
import numpy as np


from Environment import Environment
from Mission import Mission
from Robot import Robot
from Simulation import Simulation
from INS import INS
from Sonar import Sonar
from TBN_SURF_LocalizationEstimator import TBN_SURF_LocalizationEstimator
from ScanMatching_LocalizationEstimator import ScanMatching_LocalizationEstimator

# =============================================================================
# Load data
# =============================================================================

filename = "Environment_data/area_1/run_data_cubic_200x200_100x100_without_ins.pkl"

dill.load_session(filename)

# =============================================================================
# New INS
# =============================================================================

ins = INS(linear_bias = [0, 0, 0], \
          linear_white_noise_standard_deviation = [1e-3, 0, 0], \
          linear_bias_instability_standard_deviation = [0, 1e-7, 0], \
          angular_bias = [0, 0, 0], \
          angular_white_noise_standard_deviation = [0, 0, 0], \
          angular_bias_instability_standard_deviation = [0, 0, 0])

start = time.time()

robot.replace_INS_measures(0.5, ins, "Dead reckoning")

print( "Dead reckoning | Time elapsed : {} seconds.".format(time.time() - start) )

# =============================================================================
# TBN
# =============================================================================


for i in range(2, 3):
    start = time.time()
    
    TBN_Estimator = TBN_SURF_LocalizationEstimator(0.5, np.eye(3), np.eye(3), ins.getCovarianceOfMeasures(), environment, precision = i)
    
    robot.compute_new_trajectory(TBN_Estimator, name = "TBN {}".format(i))
    
    print( "TBN {} | Time elapsed : {} seconds.".format(i, time.time() - start) )

# =============================================================================
# Scan Matching
# =============================================================================

start = time.time()

ScanMatching_Estimator = ScanMatching_LocalizationEstimator(0.5, np.eye(3), np.eye(3), ins.getCovarianceOfMeasures())

robot.compute_new_trajectory(ScanMatching_Estimator, name = "ScanMatching")

print( "ScanMatching | Time elapsed : {} seconds.".format(time.time() - start) )




robot.plot_uncertainty()

robot.comparison_plot()
