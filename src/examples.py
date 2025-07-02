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


predicted_O_linear = scm_optim.predict_O(B, X, scm_optim.compute_linear_gradient, contact_subgraph)
predicted_connectome_linear = scm_optim.predict_B(X, predicted_O_linear, contact_subgraph)
residual_linear = B - predicted_connectome_linear