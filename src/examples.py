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

# Predictions for O, XOXT, and f(XOXT) for all three models. In the linear case, XOXT = f(XOXT)

predicted_O_linear = scm_optim.predict_O(B, X, contact_subgraph, scm_optim.compute_gradient)
predicted_XOXT_linear = X @ predicted_O_linear @ np.transpose(X)
predicted_connectome_linear = predicted_XOXT_linear
residual_linear = B - predicted_connectome_linear

predicted_O_relu = scm_optim.predict_O(B, X, contact_subgraph, scm_optim.compute_ReLU_gradient)
predicted_XOXT_relu = X @ predicted_O_relu @ np.transpose(X)
predicted_connectome_relu = np.maximum(0, predicted_XOXT_relu)
residual_relu = B - predicted_connectome_relu

predicted_O_sigmoid = scm_optim.predict_O(B, X, contact_subgraph, scm_optim.compute_sigmoid_gradient)
predicted_XOXT_sigmoid = X @ predicted_O_sigmoid @ np.transpose(X)
predicted_connectome_sigmoid = scm_optim.mat_sigmoid(predicted_XOXT_sigmoid)
residual_sigmoid = B - predicted_connectome_sigmoid

predicted_O_tanh = scm_optim.predict_O(B, X, contact_subgraph, scm_optim.compute_tanh_gradient)
predicted_XOXT_tanh = X @ predicted_O_tanh @ np.transpose(X)
predicted_connectome_tanh = np.tanh(predicted_XOXT_tanh)
residual_tanh = B - predicted_connectome_tanh

predicted_O_rectified_tanh = scm_optim.predict_O(B, X, contact_subgraph, scm_optim.compute_rectified_tanh_gradient)
predicted_XOXT_rectified_tanh = X @ predicted_O_rectified_tanh @ np.transpose(X)
predicted_connectome_rectified_tanh = np.maximum(0, np.tanh(predicted_XOXT_rectified_tanh))
residual_rectified_tanh = B - predicted_connectome_rectified_tanh


# min_val = predicted_O_linear.min()
# max_val = predicted_O_linear.max()
# absolute_max= max(abs(min_val), abs(max_val))
# absolute_norm = TwoSlopeNorm(vmin = -absolute_max, vcenter=0, vmax = absolute_max)
# plt.imshow(predicted_O_linear, norm=absolute_norm, cmap="bwr")
# plt.title("Linear - Predicted O")
# plt.colorbar()
# plt.show()