# -*- coding: utf-8 -*-
"""
Created on Wed May 23 15:05:24 2018

@author: fabrice.lallement
"""

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

class INS:

    def __init__(self, linear_bias = [0, 0, 0], \
                 linear_white_noise_standard_deviation = [0, 0, 0], \
                 linear_bias_instability_standard_deviation = [0, 0, 0], \
                 angular_bias = [0, 0, 0], \
                 angular_white_noise_standard_deviation = [0, 0, 0], \
                 angular_bias_instability_standard_deviation = [0, 0, 0]):
        
        self.linear_bias                                    = linear_bias
        self.linear_white_noise_standard_deviation          = linear_white_noise_standard_deviation
        self.linear_bias_instability_standard_deviation     = linear_bias_instability_standard_deviation
        
        self.angular_bias                                   = angular_bias
        self.angular_white_noise_standard_deviation         = angular_white_noise_standard_deviation
        self.angular_bias_instability_standard_deviation    = angular_bias_instability_standard_deviation

# =============================================================================
#     x_constant_bias             = 0.0;
#     x_white_noise_std           = 1e-4;
#     x_bias_instability_std      = 0.0; 
# 
#     y_constant_bias             = 0.0;
#     y_white_noise_std           = 1e-4;
#     y_bias_instability_std      = 0.0;   
# 
#     z_constant_bias             = 0.0;
#     z_white_noise_std           = 0.0;
#     z_bias_instability_std      = 0.0;
# 
#     roll_constant_bias          = 0.0;
#     roll_white_noise_std        = 0.0;
#     roll_bias_instability_std   = 0.0;
# 
#     pitch_constant_bias         = 0.0;
#     pitch_white_noise_std       = 0.0;
#     pitch_bias_instability_std  = 0.0;
# 
#     yaw_constant_bias           = 0.0;
#     yaw_white_noise_std         = 1e-6;
#     yaw_bias_instability_std    = 0.0;
# =============================================================================
        
        
        return

    def __str__(self):
        return "INS".format()
