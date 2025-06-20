import numpy as np
from numpy import linalg
import pandas as pd
import matplotlib.pyplot as plt
import sys
import time

import samp
import scm_optim
import csv

P = np.loadtxt("../data/alpha_exp_GapJunctContact.txt_ContactSubgraphMatrix.txt_100.csv", delimiter=",")
B = np.loadtxt("../data/GapJunctContact.csv", delimiter=",") # connectome; which neurons are synapsed - 185 x 185
X = np.loadtxt("../data/INXExpressionJustContact.csv", delimiter=",") #protein expression of every neuron - should be 185 X 18?
contact_subgraph = np.loadtxt("../data/ContactSubgraphMatrix.csv", delimiter=",", ) #which neurons are physically touching

num_samples = 1000
error_norms = []
O_samples = []

start_time = time.time()

for i in range(num_samples):
	predicted_O = scm_optim.predict_O(B, X, contact_subgraph, scm_optim.compute_gradient)
	O_samples.append(predicted_O)
	error_norm = linalg.norm(B - np.maximum(0, np.tanh(X @ predicted_O @ np.transpose(X))))
	error_norms.append(error_norm)

#Timing
end_time = time.time()
time_elapsed = end_time - start_time
print("time elapsed: " + str(time_elapsed))

#Mean O
stacked_Os = np.stack(O_samples)
average_O = np.mean(stacked_Os, axis = 0)

#Error statistics
error_norms_np = np.array(error_norms)
mean_err = np.mean(error_norms_np)
std_err = np.std(error_norms_np)
print("Mean error: " + str(mean_err))
print("Std deviation of error: " + str(std_err))

#Saving files
# np.savetxt("../output/predicted_rules/predicted_rules_tanh_100sampleavg.csv", average_O, delimiter=",") # EDIT FOR DIFFERENT MODELS