import numpy as np
from numpy import linalg
import pandas as pd
import matplotlib.pyplot as plt
import sys
import time

import samp
import scm_optim
import csv

B = np.loadtxt("../data/GapJunctContact.csv", delimiter=",") # connectome; which neurons are synapsed - 185 x 185
X = np.loadtxt("../data/INXExpressionJustContact.csv", delimiter=",") #protein expression of every neuron - should be 185 X 18?
contact_subgraph = np.loadtxt("../data/ContactSubgraphMatrix.csv", delimiter=",", ) 

tols = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512, 1.024, 2.048, 4.096]

avg_errors = []

sample_count = 10
for tol in tols:
	sum_error = 0
	counter = 0
	for i in range(sample_count):
		predicted_O = scm_optim.predict_O(B, X, scm_optim.compute_linear_gradient, C = contact_subgraph, tol = tol)
		predicted_B = scm_optim.predict_B_linear(X, predicted_O, contact_subgraph)
		sum_error += linalg.norm(B - predicted_B)
	avg_errors.append(sum_error / sample_count)
	print("Finished samples for threshold " + str(tol))
	print(sum_error / sample_count)

print(avg_errors)