import time

import numpy as np

from Environment import Environment
from Mission import Mission
from Robot import Robot
from Simulation import Simulation
from INS import INS
from Sonar import Sonar
from Matcher import Matcher

import dill

import cv2

filename = 'run_data_cubic_200x200_100x100_without_noise.pkl'

dill.load_session(filename)