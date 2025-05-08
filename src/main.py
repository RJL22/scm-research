import numpy as np
from numpy import linalg
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import sys

import samp
import scm_optim
import csv


P = np.loadtxt("../data/alpha_exp_GapJunctContact.txt_ContactSubgraphMatrix.txt_100.csv", delimiter=",")
B = np.loadtxt("../data/GapJunctContact.csv", delimiter=",") # connectome; which neurons are synapsed - 185 x 185
X = np.loadtxt("../data/INXExpressionJustContact.csv", delimiter=",") #protein expression of every neuron - should be 185 X 18?
contact_subgraph = np.loadtxt("../data/ContactSubgraphMatrix.csv", delimiter=",", ) #which neurons are physically touching

#Randomly sampling B matrices and generating O matrices
B_samples = []
O_samples = []

sample_count = 2000

# predicted_O = scm_optim.predict_O_sigmoid(B, X, contact_subgraph)


# We want to visualize: O, XOXT, and f(XOXT) for all three models. In the linear case, XOXT = f(XOXT)


predicted_O_linear = scm_optim.predict_O(B, X, contact_subgraph)
predicted_XOX_linear = X @ predicted_O_linear @ np.transpose(X)
predicted_connectome_linear = predicted_XOX_linear

predicted_O_relu = scm_optim.predict_O_ReLU(B, X, contact_subgraph)
predicted_XOX_relu = X @ predicted_O_relu @ np.transpose(X)
predicted_connectome_relu = np.maximum(0, predicted_XOX_relu)

predicted_O_sigmoid = scm_optim.predict_O_sigmoid(B, X, contact_subgraph)
predicted_XOX_sigmoid = X @ predicted_O_sigmoid @ np.transpose(X)
predicted_connectome_sigmoid = scm_optim.mat_sigmoid(predicted_XOX_sigmoid)

# global_min = min(predicted_XOX_linear.min(), predicted_XOX_relu.min(), predicted_XOX_sigmoid.min())
# global_max = max(predicted_XOX_linear.max(), predicted_XOX_relu.max(), predicted_XOX_sigmoid.max())
# global_extreme = max(abs(global_min), abs(global_max))
# absolute_norm = TwoSlopeNorm(vmin=-global_extreme, vcenter=0, vmax=global_extreme)

plt.imshow(predicted_connectome_sigmoid, cmap="bwr")              #Change
plt.title("Sigmoid Model - Predicted Connectome")                            #Change
plt.colorbar()


plt.savefig('../figures/predicted_connectome_sigmoid.png')                          #Change

plt.show()

# np.savetxt("relu_z_scores_2000_samples.csv", z_scores, delimiter=",")



"""

for i in range(sample_count):
	B_samples.append(samp.generate_B(P))
for i in range(sample_count):
	if i % 100 == 0:
		print("Finished 100 samples")
	O_samples.append(np.matrix(scm_optim.predict_O_ReLU(B_samples[i], X, contact_subgraph)))

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
predicted_O = scm_optim.predict_O_ReLU(B, X, contact_subgraph)

#Computing O z scores
z_scores = (predicted_O - mean_O) / sd_O


"""

# error = linalg.norm(B - np.maximum(0, X @ predicted_O @ X.T))
# error = linalg.norm(B - scm_optim.mat_sigmoid(X @ predicted_O @ X.T))
# print("Frobenius Hi error: " + str(error))
# b_norm = linalg.norm(B)
# print("B norm: " + str(b_norm))


# plt.imshow(predicted_O, cmap="bwr")
# plt.title("O Matrix")
# plt.colorbar()
# plt.show()


# plt.savefig('PNAS_z_scores_2000_samples.png')
# np.savetxt("relu_z_scores_2000_samples.csv", z_scores, delimiter=",")