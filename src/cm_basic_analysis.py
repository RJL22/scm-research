import numpy as np
from numpy import linalg
import pandas as pd
import matplotlib.pyplot as plt
import sys
import time

import samp
import scm_optim
import csv

B = np.loadtxt("../data/CM_nophy_B_certain_enriched (1).txt", delimiter=",")
X = np.loadtxt("../data/CM_nophy_X_certain_enriched (1).txt", delimiter=",")

# predicted_O_linear = scm_optim.predict_O(B, X, scm_optim.compute_linear_gradient, tol=5)
# predicted_B_linear = X @ predicted_O_linear @ np.transpose(X)

# predicted_O_ReLU = scm_optim.predict_O(B, X, scm_optim.compute_ReLU_gradient, tol=5)
# predicted_B_ReLU = np.maximum(0, X @ predicted_O_ReLU @ np.transpose(X))

# predicted_O_tanh = scm_optim.predict_O(B, X, scm_optim.compute_tanh_gradient, tol=10)
# predicted_B_tanh = np.tanh(X @ predicted_O_tanh @ np.transpose(X))

# predicted_O_rectified_tanh = scm_optim.predict_O(B, X, scm_optim.compute_rectified_tanh_gradient, tol=0.03)
# predicted_B_rectified_tanh = np.maximum(0, np.tanh(X @ predicted_O_rectified_tanh @ np.transpose(X)))

# print(linalg.norm(B - predicted_B_linear))
# print(linalg.norm(B - predicted_B_ReLU))
# print(linalg.norm(B - predicted_B_tanh))
# print(X.shape)

gj_B = np.loadtxt("../data/GapJunctContact.csv", delimiter=",") # connectome; which neurons are synapsed - 185 x 185
# gj_X = np.loadtxt("../data/INXExpressionJustContact.csv", delimiter=",") #protein expression of every neuron - should be 185 X 18?
# contact_subgraph = np.loadtxt("../data/ContactSubgraphMatrix.csv", delimiter=",", ) #which neurons are physically touching 
# predicted_O_ReLU = scm_optim.predict_O(gj_B, gj_X, scm_optim.compute_linear_gradient, tol=5)
contact_subgraph = np.loadtxt("../data/ContactSubgraphMatrix.csv", delimiter=",", ) #which neurons are physically touching
print(np.count_nonzero(contact_subgraph))