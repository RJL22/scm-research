import numpy as np
import copy
from numpy import linalg

import samp


# Compute the gradient of the objective function with respect to O, applying M_B and M_O if provided
def compute_gradient(B, X, O, M_B=None, M_O=None, regularization_factor=0):
    error = B - X @ O @ X.T
    
    # Apply mask M_B to the error if provided
    if M_B is not None:
        error = M_B * error
    
    gradient = -2 * X.T @ error @ X

    # Apply mask M_O to the gradient if provided
    if M_O is not None:
        gradient = M_O * gradient
    
    return gradient + (regularization_factor * O)

#Compute the gradient of the ReLU objective function with respect to O, applying M_B and M_O if provided
def compute_ReLU_gradient(B, X, O, M_B=None, M_O=None, regularization_factor=0):
    #Let L = Loss
    #Let Z = XOXT

    Z = X @ O @ X.T

    M = (Z > 0).astype(int)
    # M = np.sign(Z)

    dLdZ = 2 * ((np.maximum(0, Z)) - B) * M

    gradient = X.T @ dLdZ @ X  #dLdO

    # Apply mask M_O to the gradient if provided
    if M_O is not None:
        gradient = M_O * gradient

    return gradient + (regularization_factor * O)

#Function for computing sigmoid of a matrix elementwise
def mat_sigmoid(M):
    return np.ones_like(M) / (np.ones_like(M) + np.exp(-M))


def compute_sigmoid_gradient(B, X, O, M_B=None, M_O=None, regularization_factor=0):
    #Let L = Loss
    #Let Z = XOXT

    Z = X @ O @ X.T

    M = mat_sigmoid(Z) * (1 - mat_sigmoid(Z))
    # M = np.sign(Z)

    dLdZ = 2 * (mat_sigmoid(Z) - B) * M

    gradient = X.T @ dLdZ @ X  #dLdO

    # Apply mask M_O to the gradient if provided
    if M_O is not None:
        gradient = M_O * gradient

    return gradient + (regularization_factor * O)

def adam_optimizer(B, X, O_init=None, M_B=None, M_O=None, regularization_factor=0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iters=1000, tol=1e-4):
    """Perform optimization using Adam."""
    if O_init is None:
        O = initialize_O(X)
    else:
        O = copy.copy(O_init)

    m = np.zeros_like(O)  # Initialize 1st moment vector
    v = np.zeros_like(O)  # Initialize 2nd moment vector
    t = 0  # Initialize timestep

    for i in range(max_iters):
        t += 1
        gradient = compute_gradient(B, X, O, M_B, M_O, regularization_factor=regularization_factor)
        
        # Check for convergence
        if np.linalg.norm(gradient) < tol:
            print(f"Convergence reached at iteration {i}")
            break

        # Apply Adam updates
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient ** 2)
        
        # Bias-corrected moment estimates
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Compute the new O with updates
        O = O - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
        # Project O values to the range [0, 1] # ! ensure positivity
        # O = np.clip(O, 0, 1)
    
    return O

#CHange tol ()
# Plot weight of the rule vs the error of the rule (for both versions)
# Check other nonlinearities to see if there are generic rules (rules that are found in all models)
def adam_optimizer_ReLU(B, X, O_init=None, M_B=None, M_O=None, regularization_factor=0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iters=1000, tol=1e-4):
    """Perform optimization using Adam."""
    if O_init is None:
        O = initialize_O(X)
    else:
        O = copy.copy(O_init)

    m = np.zeros_like(O)  # Initialize 1st moment vector
    v = np.zeros_like(O)  # Initialize 2nd moment vector
    t = 0  # Initialize timestep

    for i in range(max_iters):
        t += 1
        gradient = compute_ReLU_gradient(B, X, O, M_B, M_O)
        
        # Check for convergence
        if np.linalg.norm(gradient) < tol:
            print(f"Convergence reached at iteration {i}")
            break

        # Apply Adam updates
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient ** 2)
        
        # Bias-corrected moment estimates
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Compute the new O with updates
        O = O - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
        # Project O values to the range [0, 1] # ! ensure positivity
        # O = np.clip(O, 0, 1)
    
    return O

# Check other nonlinearities to see if there are generic rules (rules that are found in all models)
def adam_optimizer_sigmoid(B, X, O_init=None, M_B=None, M_O=None, regularization_factor=0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iters=1000, tol=1e-3):
    """Perform optimization using Adam."""
    if O_init is None:
        O = initialize_O(X)
    else:
        O = copy.copy(O_init)

    m = np.zeros_like(O)  # Initialize 1st moment vector
    v = np.zeros_like(O)  # Initialize 2nd moment vector
    t = 0  # Initialize timestep

    for i in range(max_iters):
        t += 1
        gradient = compute_sigmoid_gradient(B, X, O, M_B, M_O)
        
        # Check for convergence
        if np.linalg.norm(gradient) < tol:
            print(f"Convergence reached at iteration {i}")
            break

        # Apply Adam updates
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient ** 2)
        
        # Bias-corrected moment estimates
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Compute the new O with updates
        O = O - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        #Debugging to see error for each iteration
        # error = linalg.norm(B - mat_sigmoid(X @ O @ X.T))
        # print("Frobenius Hi error: " + str(error))
        
        # Project O values to the range [0, 1] # ! ensure positivity
        # O = np.clip(O, 0, 1)
    
    return O

#Uses the adam optimizer to find O matrix that minimizes loss. Regularization factor is currently hard-coded to 0.
def predict_O(B, X, C):
	dim = X.shape[1]
	O = samp.initialize_O(dim)
	O = adam_optimizer(B, X, O, M_B=C, regularization_factor=0.0)
	return O


def predict_O_ReLU(B, X, C, reg_factor=0):
    dim = X.shape[1]
    O = samp.initialize_O(dim)
    O = adam_optimizer_ReLU(B, X, O, M_B=C, regularization_factor=reg_factor)
    return O

def predict_O_sigmoid(B, X, C, reg_factor=0):
    dim = X.shape[1]
    O = samp.initialize_O(dim)
    O = adam_optimizer_sigmoid(B, X, O, M_B=C, regularization_factor=reg_factor)
    return O

#Original O prediction from PNAS paper
def getRuleMatrix(B, X, C, alpha):
    #Setup
    K = np.kron(X, X)

    # #Zeroing out duplicates (lower left triangle of matrix of contactome)
    # for i in range(len(C)):
    #   for j in range(i):
    #       C[i, j] = 0

    contactome_vector = C.flatten()

    #Shortening Kronecker product and connectome vector
    K_prime = np.delete(K,np.nonzero(1-contactome_vector),0)
    K_prime_transpose = np.transpose(K_prime)
    connectome_vector = B.flatten()
    connectome_vector_prime = connectome_vector[contactome_vector==1]


    #Calculation
    K_prime_cross_alpha = np.linalg.pinv(K_prime_transpose @ K_prime + (alpha * np.identity(K_prime.shape[1]))) @ K_prime_transpose
    o = K_prime_cross_alpha @ connectome_vector_prime

    return o.reshape(X.shape[1], X.shape[1])
