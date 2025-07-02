import numpy as np
from numpy import linalg
import pandas as pd
import matplotlib.pyplot as plt
import sys

import samp
import scm_optim
import csv

#Loading raw data
P = np.loadtxt("../data/alpha_exp_GapJunctContact.txt_ContactSubgraphMatrix.txt_100.csv", delimiter=",")
B = np.loadtxt("../data/GapJunctContact.csv", delimiter=",") # connectome; which neurons are synapsed - 185 x 185
X = np.loadtxt("../data/INXExpressionJustContact.csv", delimiter=",") #protein expression of every neuron - should be 185 X 18?
contact_subgraph = np.loadtxt("../data/ContactSubgraphMatrix.csv", delimiter=",", ) #which neurons are physically touching

#Generating predicted O matrices
sample_count = 1000 
O_samples = []

for i in range(sample_count):
	if i % 100 == 0:
		print("Finished " + str(i) + "samples")
	O_samples.append(scm_optim.predict_O(samp.generate_B(P), X, scm_optim.compute_rectified_itanh_gradient, C = contact_subgraph, tol = 0.5))

#Computing mean, variance, and sd of rules from randomized connectomes
stacked_Os = np.stack(O_samples)
mean_O = np.mean(stacked_Os, axis=0)
sd_O = np.std(stacked_Os, axis=0)

#Importing mean rule matrix from true connectome
predicted_rules = np.loadtxt("../output/predicted_rules/predicted_rules_rectified_itanh_100sampleavg.csv", delimiter=",")

#Computing z-scores
#z_scores = (predicted_rules - mean_O) / sd_O
print(sd_O)
#Saving data
np.savetxt("../output/statistics/randomizedB_1000samples_meanO_rectified_itanh.csv", mean_O, delimiter=",")
np.savetxt("../output/statistics/randomizedB_1000samples_sdO_rectified_itanh.csv", sd_O, delimiter=",")

