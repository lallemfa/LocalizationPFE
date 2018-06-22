# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 11:14:35 2018

@author: fabrice.lallement
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close()

a = 2 + 0.1*np.random.randn(100000)
b = np.cos(a)

plt.figure("Test")
plt.hist(a, label="Original")
plt.hist(b, label="Cos")
plt.legend()

mean_a = np.mean(a)
mean_b = np.mean(b)

std_a = np.std(a)
std_b = np.std(b)

print("{} | {}".format(std_a, std_b))
print("{} | {}".format(mean_a, mean_b))