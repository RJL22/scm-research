import math
import numpy as np
from numpy import linalg
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import sys

import samp
import scm_optim
import csv

def graph_O(O, title): 
	min_val = O.min()
	max_val = O.max()
	# absolute_max= max(abs(min_val), abs(max_val))
	# absolute_norm = TwoSlopeNorm(vmin = -absolute_max, vcenter=0, vmax = absolute_max)
	# plt.imshow(O, norm=absolute_norm, cmap="bwr")
	plt.imshow(O, cmap="Greys")
	plt.title(title)
	# plt.colorbar()
	plt.savefig("../output/figures/significant_rules_rectified_itanh.png")
	plt.show()


# O_tanh = np.loadtxt("../output/predicted_rules/predicted_rules_rectified_tanh_100sampleavg.csv", delimiter=",")
# graph_O(O_tanh, "Average Predicted O - Rectified Tanh")

def plot_nonlinearity(f):
	x = np.linspace(-10, 10, 500)
	
	y = [f(i) for i in x]

	plt.plot(x, y, label='Tanh', color='blue')
	plt.title('ReTanh(x^0.5)')
	plt.xlabel('x')
	plt.ylabel('ReTanh(x^0.5)')
	plt.grid(False)
	plt.axhline(0, color='black', linewidth=0.5)
	plt.axvline(0, color='black', linewidth=0.5)
	plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
	plt.xlim(-5, 5)
	plt.ylim(-5, 5)
	#plt.savefig("../output/figures/significant_rules_rectified_tanh.png")
	plt.show()


def re_ztanh(x):
	if x <= 0:
		return 0
	else:
		return math.tanh(x ** (1/2))