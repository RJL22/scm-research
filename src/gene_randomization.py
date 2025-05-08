import numpy as np
import pandas as pd
import sys

import samp
import scm_optim

#Loading in data
sys.path.append("/Users/ryanlee/Documents/Kovacs/SCM")

P = np.loadtxt("/Users/ryanlee/Documents/Kovacs/SCM/alpha_exp_GapJunctContact.txt_ContactSubgraphMatrix.txt_100.csv", delimiter=",")
B = np.loadtxt("/Users/ryanlee/Documents/Kovacs/SCM/GapJunctContact.csv", delimiter=",") # connectome; which neurons are synapsed - 185 x 185
X = np.loadtxt("/Users/ryanlee/Documents/Kovacs/SCM/INXExpressionJustContact.csv", delimiter=",") #protein expression of every neuron - should be 185 X 18?
contact_subgraph = np.loadtxt("/Users/ryanlee/Documents/Kovacs/SCM/ContactSubgraphMatrix.csv", delimiter=",", ) #which neurons are physically touching

#Randomly sampling X matrices and generating O matrices
X_samples = []
O_samples = []

sample_count = 5

#Generating shuffled gene samples
for i in range(sample_count):
	X_samples.append(X[:, np.random.permutation(X.shape[1])])

for i in range(sample_count):
	O_samples.append(scm_optim.predict_O(B, X_samples[i], contact_subgraph))

#Initializing O mean and variance matrices
mean_O = np.zeros(O_samples[0].shape)
var_O = np.zeros(O_samples[0].shape)


#Computing O mean
for i in range(sample_count):
	mean_O += O_samples[i]

mean_O = mean_O / sample_count

#Computing O variance
for i in range(sample_count):
	var_O += np.square(O_samples[i] - mean_O)

var_O = var_O / (sample_count - 1)

#Computing O standard deviation matrix
sd_O = np.sqrt(var_O)

#Computing O matrix from true B
# predicted_O = scm_optim.predict_O(B, X, contact_subgraph)
predicted_O = scm_optim.getRuleMatrix(B, X, contact_subgraph, 0)

#Computing O z scores
z_scores = (predicted_O - mean_O) / sd_O

# print(z_scores.shape)

sig_count = 0
for i in range(18):
	for j in range(18):
		if abs(z_scores[i, j]) > 2:
			# print("Significant rule at: " + str(i) + ", " + str(j))
			print(z_scores[i, j])
			sig_count += 1
print("Sig count: " + str(sig_count))