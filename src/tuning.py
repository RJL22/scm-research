import numpy as np
from numpy import linalg
import pandas as pd
import matplotlib.pyplot as plt
import sys
import time

import samp
import scm_optim
import csv

# B = np.loadtxt("../data/CM_nophy_B_certain_enriched (1).txt", delimiter=",")
# X = np.loadtxt("../data/CM_nophy_X_certain_enriched (1).txt", delimiter=",")
B = np.loadtxt("../data/GapJunctContact.csv", delimiter=",") # connectome; which neurons are synapsed - 185 x 185
X = np.loadtxt("../data/INXExpressionJustContact.csv", delimiter=",") #protein expression of every neuron - should be 185 X 18?
contact_subgraph = np.loadtxt("../data/ContactSubgraphMatrix.csv", delimiter=",", ) 

tols = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512, 1.024, 2.048, 4.096]
rectified_ztanh_tols = [0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512, 1.024, 2.048, 4.096]
rectified_itanh_tols = [0.256, 0.512, 1.024, 2.048, 4.096]
"""
avg_errors = []

sample_count = 10
for tol in rectified_itanh_tols:
	sum_error = 0
	counter = 0
	for i in range(sample_count):
		# sum_error += linalg.norm(contact_subgraph * (B - X @ scm_optim.predict_O(B, X, scm_optim.compute_linear_gradient, C = contact_subgraph, tol = tol) @ np.transpose(X)))
		# sum_error += linalg.norm(contact_subgraph * (B - np.maximum(0, X @ scm_optim.predict_O(B, X, scm_optim.compute_ReLU_gradient, C = contact_subgraph, tol = tol) @ np.transpose(X))))
		#sum_error += linalg.norm(contact_subgraph * (B - np.tanh(X @ scm_optim.predict_O(B, X, scm_optim.compute_tanh_gradient, C = contact_subgraph, tol = tol) @ np.transpose(X))))
		predicted_O = scm_optim.predict_O(B, X, scm_optim.compute_rectified_itanh_gradient, C = contact_subgraph, tol = tol)
		predicted_B = scm_optim.predict_B_rectified_itanh(X, predicted_O, contact_subgraph)
		sum_error += linalg.norm(B - predicted_B)
		# (predicted_O, convergence) = scm_optim.predict_O_testing(B, X, scm_optim.compute_rectified_tanh_gradient, C = contact_subgraph, tol = tol)
		# if convergence is True:
		# 	counter += 1
		# sum_error += linalg.norm(contact_subgraph * (B - np.tanh(np.maximum(0, X @ predicted_O @ np.transpose(X)))))
	# print("Convergence count: " + str(counter))
	avg_errors.append(sum_error / sample_count)
	print("Finished samples for threshold " + str(tol))
	print(sum_error / sample_count)

print(avg_errors)

# plt.scatter(tols, avg_errors)
# plt.xscale("log")
# plt.xlabel("Gradient Threshold")
# plt.ylabel("Average Error Norm")
# plt.title("Average Error Norm vs. Gradient Threshold - Rectified Tanh") ##### Modify for different models
# plt.savefig("../output/tuning/gradthreshold_vs_avgerror_10sample_rectified_tanh2.png") ##### Modify for different models
# plt.show()
"""

linear_avg_errors = [31.787380465416454, 31.787380540087973, 31.787380838129245, 31.78738202536426, 31.78738662436221, 31.787403193927588, 31.78745388006822, 31.78759378585974, 31.788022911453766, 31.789420637411393, 31.794099762940675, 31.808803200599222, 31.848937378323456]
ReLU_avg_errors = [31.230527509676477, 31.239033509144853, 31.242596660818258, 31.242426938079888, 31.242121086627556, 31.235623475473624, 31.242024153469963, 31.23838450207473, 31.243692419195106, 31.23934192531724, 31.259245008394203, 31.290884644246564, 31.38597881843456]
tanh_avg_errors = [31.77883165072992, 31.778843470089118, 31.778864754212663, 31.778903404988643, 31.778986600552237, 31.779153820957482, 31.77950253159717, 31.780313638157196, 31.782017724146375, 31.785003478270074, 31.791720564335243, 31.80902286945416, 31.84703789064746]
rectified_tanh_avg_errors = [31.182360060179395, 31.185113945124424, 31.18697001337862, 31.18705299641407, 31.18825748767364, 31.18713213102285, 31.191082105001545, 31.192412802530146, 31.195507241756992, 31.20687273921042, 31.235703200028876, 31.29611976443896, 31.39682877569396]
rectified_ztanh_avg_errors = [31.919537806825126, 31.905504632945032, 31.921612079863802, 31.954398652529612, 31.96398327536931, 31.95978726688647, 31.959475844222617, 31.96145579676583, 32.0215157303396, 32.141975321605926]
rectified_itanh_avg_errors = [34.386231518223724, 34.392349689823824, 34.41024208207125, 34.446107195304634, 34.51745106035959]
# plt.scatter(tols, linear_avg_errors, label="Linear", marker = "x")
# plt.scatter(tols, ReLU_avg_errors, label = "ReLU", marker = "x")
# plt.scatter(tols, tanh_avg_errors, label = "Tanh", marker = "x")
# plt.scatter(tols, rectified_tanh_avg_errors, label = "Rectified Tanh", marker="x")
# plt.scatter(rectified_ztanh_tols, rectified_ztanh_avg_errors, label = "Rectified Tanh(x^2)", marker="x")
plt.scatter(rectified_itanh_tols, rectified_itanh_avg_errors, marker="x")
plt.xscale("log")
plt.xlabel("Gradient Threshold")
plt.ylabel("Average Error Norm")
plt.title("Average Error Norm vs. Gradient Threshold")
plt.legend()
# plt.savefig("../output/tuning/gradthreshold_vs_avgerror_10sample_v2.png")
plt.show()

