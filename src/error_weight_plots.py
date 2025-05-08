import numpy as np
from numpy import linalg
import pandas as pd
import matplotlib.pyplot as plt
import sys

import samp
import scm_optim
import csv

sd = np.loadtxt("relu_sdO_on_trueB_2000_samples.csv", delimiter=",")
o = np.loadtxt("relu_meanO_on_trueB_2000_samples.csv", delimiter=",")
# sd = np.loadtxt("linear_gd_sdO_on_trueB_2000_samples.csv", delimiter=",")
# o = np.loadtxt("linear_gd_meanO_on_trueB_2000_samples.csv", delimiter=",")

sd_flat = sd.flatten()
o_flat = o.flatten()

# Plot the points
plt.scatter(o_flat, sd_flat)
plt.xlabel('Mean')
plt.ylabel('SD')
plt.title('Scatter plot of points from matrices')
plt.grid(True)
plt.show()