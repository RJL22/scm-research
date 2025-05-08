import numpy as np

#Given a matrix P whose entries are probabilities of synapse formation, randomly samples and returns B (connectome)
def generate_B(P):
	B = np.zeros_like(P)
	for row_num in range(len(P)):
		for col_num in range(len(P[0])):
			B[row_num][col_num] = np.random.binomial(1, P[row_num][col_num])
	return B

#Initializes a random square (dim x dim) O matrix 
def initialize_O(dim):
    return np.random.rand(dim, dim) * 0.01