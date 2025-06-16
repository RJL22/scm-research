import numpy as np
from numpy import linalg
import pandas as pd
import matplotlib.pyplot as plt
import sys

import samp
import scm_optim
import csv

B = np.loadtxt("../data/GapJunctContact.csv", delimiter=",") # connectome; which neurons are synapsed - 185 x 185
X = np.loadtxt("../data/INXExpressionJustContact.csv", delimiter=",") #protein expression of every neuron - should be 185 X 18?
contact_subgraph = np.loadtxt("../data/ContactSubgraphMatrix.csv", delimiter=",", ) #which neurons are physically touching


# pnas_mean_O = np.loadtxt("/Users/ryanlee/Documents/Kovacs/SCM/rule_matrix_0.2150rnd_av.csv", delimiter=",") #protein expression of every neuron - should be 185 X 18?
# pnas_variance_O = np.loadtxt("/Users/ryanlee/Documents/Kovacs/SCM/rule_matrix_0.2150rnd_var.csv", delimiter=",")
# pnas_sd = np.sqrt(pnas_variance_O)
# pnas_predicted_O = scm_optim.getRuleMatrix(B, X, contact_subgraph, 0.215)
# pnas_z_scores = (pnas_predicted_O - pnas_mean_O) / pnas_sd
# pnas_significant_rules = pnas_z_scores
# pnas_significant_rules[pnas_significant_rules < 2] = 0
# pnas_significant_rules[pnas_significant_rules >= 2] = 1
#print(predicted_O)


# z_scores = np.loadtxt('relu_z_scores_2000_samples.csv', delimiter=',')

#Generating predicted O matrices
O_samples = []

sample_count = 1000

for i in range(sample_count):
	if i % 100 == 0:
		print("Finished 100 samples")
	O_samples.append(np.matrix(scm_optim.predict_O_ReLU(B, X, contact_subgraph, reg_factor=0.1)))
	# O_samples.append(scm_optim.getRuleMatrix(B_samples[i], X, contact_subgraph, 0.225))

#Initializing O mean and variance matrices
mean_O = np.zeros(O_samples[0].shape)
var_O = np.zeros(O_samples[0].shape)

# #Computing O mean
for i in range(sample_count):
	mean_O += O_samples[i]

mean_O = mean_O / sample_count

# #Computing O variance
for i in range(sample_count):
	var_O += np.square(O_samples[i] - mean_O)

var_O = var_O / (sample_count - 1)

# #Computing O standard deviation matrix
sd_O = np.sqrt(var_O)


#Computing O matrix from true B
# predicted_O = scm_optim.predict_O_ReLU(B, X, contact_subgraph)
# predicted_O = scm_optim.getRuleMatrix(B, X, contact_subgraph, 0.225)

#Computing O z scores
# z_scores = (predicted_O - mean_O) / sd_O

# print("Z-score: ")
# print(z_scores)

np.savetxt("../statistics/regularized_ReLU_meanO_on_trueB_2000_samples.csv", mean_O, delimiter=",")
np.savetxt("../statistics/regularized_ReLU_sdO_on_trueB_2000_samples.csv", sd_O, delimiter=",")



linear_O_test_sd = np.loadtxt("linear_gd_sdO_on_trueB_2000_samples.csv", delimiter=",")
linear_O_sd = np.loadtxt("/Users/ryanlee/Documents/Projects/scm-research/src/linear_gd_sdO_on_trueB_2000_samples.csv", delimiter=",")
ReLU_O_sd = np.loadtxt("/Users/ryanlee/Documents/Projects/scm-research/src/ReLU_sdO_on_trueB_2000_samples.csv", delimiter=",")

linear_O_sd_average = np.mean(linear_O_sd)
ReLU_O_sd_average = np.mean(ReLU_O_sd)
print(linear_O_sd_average)
print(ReLU_O_sd_average)
