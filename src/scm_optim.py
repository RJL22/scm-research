import numpy as np
import copy
from numpy import linalg

import samp


# Compute the gradient of the linear objective function with respect to O, applying M_B and M_O if provided
def compute_linear_gradient(B, X, O, M_B=None, M_O=None, regularization_factor=0):
    error = B - X @ O @ X.T
    
    # Apply mask M_B to the error if provided
    if M_B is not None:
        error = M_B * error
    
    gradient = -2 * X.T @ error @ X

    # Apply mask M_O to the gradient if provided
    if M_O is not None:
        gradient = M_O * gradient
    
    return gradient + (regularization_factor * np.abs(O))

#Compute the gradient of the ReLU objective function with respect to O, applying M_B and M_O if provided
def compute_ReLU_gradient(B, X, O, M_B=None, M_O=None, regularization_factor=0):
    #Let L = Loss
    #Let Z = XOXT

    Z = X @ O @ X.T

    M = (Z > 0).astype(int)

    dLdZ = 2 * ((np.maximum(0, Z)) - B) * M

    if M_B is not None:
        dLdZ = M_B * dLdZ 

    gradient = X.T @ dLdZ @ X  #dLdO

    # Apply mask M_O to the gradient if provided
    if M_O is not None:
        gradient = M_O * gradient

    return gradient + (regularization_factor * np.abs(O))

#Function for computing sigmoid of a matrix elementwise
def mat_sigmoid(M):
    return np.ones_like(M) / (np.ones_like(M) + np.exp(-M))


def compute_sigmoid_gradient(B, X, O, M_B=None, M_O=None, regularization_factor=0):
    Z = X @ O @ X.T

    M = mat_sigmoid(Z) * (1 - mat_sigmoid(Z))
    # M = np.sign(Z)

    dLdZ = 2 * (mat_sigmoid(Z) - B) * M

    if M_B is not None:
        dLdZ = M_B * dLdZ 

    gradient = X.T @ dLdZ @ X  #dLdO

    # Apply mask M_O to the gradient if provided
    if M_O is not None:
        gradient = M_O * gradient

    return gradient + (regularization_factor * np.abs(O))

def compute_tanh_gradient(B, X, O, M_B=None, M_O=None, regularization_factor=0):
    Z = X @ O @ X.T

    M = np.ones_like(Z) - np.square(np.tanh(Z))
    # M = np.sign(Z)

    dLdZ = 2 * (np.tanh(Z) - B) * M

    if M_B is not None:
        dLdZ = M_B * dLdZ 
    
    gradient = X.T @ dLdZ @ X  #dLdO

    # Apply mask M_O to the gradient if provided
    if M_O is not None:
        gradient = M_O * gradient

    return gradient + (regularization_factor * np.abs(O))

def compute_rectified_tanh_gradient(B, X, O, M_B=None, M_O=None, regularization_factor=0):
    Z = X @ O @ X.T

    M1 = (Z > 0).astype(int)
    M2 = np.ones_like(Z) - np.square(np.tanh(Z))
    M = M1 * M2

    dLdZ = 2 * (np.tanh(Z) - B) * M

    if M_B is not None:
        dLdZ = M_B * dLdZ 
    
    gradient = X.T @ dLdZ @ X  #dLdO

    # Apply mask M_O to the gradient if provided
    if M_O is not None:
        gradient = M_O * gradient

    return gradient + (regularization_factor * np.abs(O))

#rectified tanh of x^2
def compute_rectified_ztanh_gradient(B, X, O, M_B=None, M_O=None, regularization_factor=0):
    Z = X @ O @ X.T

    M1 = (Z > 0).astype(int)
    M2 = np.ones_like(Z) - np.square(np.tanh(Z ** 2))
    M = M1 * M2 * (2 * Z)

    dLdZ = 2 * (np.tanh(Z) - B) * M

    if M_B is not None:
        dLdZ = M_B * dLdZ 
    
    gradient = X.T @ dLdZ @ X  #dLdO

    # Apply mask M_O to the gradient if provided
    if M_O is not None:
        gradient = M_O * gradient

    return gradient + (regularization_factor * np.abs(O))

#rectified tanh of x^(1/2)
def compute_rectified_itanh_gradient(B, X, O, M_B=None, M_O=None, regularization_factor=0):
    Z = X @ O @ X.T
    M1 = (Z > 0).astype(int)
    M2 = np.ones_like(Z) - np.square(np.tanh(np.abs(Z) ** 0.5))
    
    M3 = Z.copy()
    mask = Z > 0
    M3[mask] = (0.5 / (M3[mask] ** 0.5))

    M = M1 * M2 * M3

    dLdZ = 2 * (np.tanh(Z) - B) * M

    if M_B is not None:
        dLdZ = M_B * dLdZ 
    
    gradient = X.T @ dLdZ @ X  #dLdO

    # Apply mask M_O to the gradient if provided
    if M_O is not None:
        gradient = M_O * gradient

    return gradient + (regularization_factor * np.abs(O))


def adam_optimizer(B, X, O_init, gradient_func, M_B=None, M_O=None, regularization_factor=0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iters=50000, tol=100):
    """Perform optimization using Adam."""
    O = None
    
    if O_init is None:
        O = initialize_O(X)
    else:
        O = copy.copy(O_init)

    m = np.zeros_like(O)  # Initialize 1st moment vector
    v = np.zeros_like(O)  # Initialize 2nd moment vector
    t = 0  # Initialize timestep
    for i in range(max_iters):
        t += 1
        gradient = gradient_func(B, X, O, M_B, M_O, regularization_factor=regularization_factor)
        
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
    
    if np.linalg.norm(gradient) > tol:
        print("Convergence not reached")

    return O


#Uses the adam optimizer to find O matrix that minimizes loss. Regularization factor is currently hard-coded to 0.
def predict_O(B, X, gradient_func, C=None, regularization_factor=0, tol=0.1):
	dim = X.shape[1]
	O = samp.initialize_O(dim)
	O = adam_optimizer(B, X, O, gradient_func, M_B=C, regularization_factor=regularization_factor, tol=tol)
	return O




#Adam optimizer that returns useful information for testing/tuning
def adam_optimizer_testing(B, X, O_init, gradient_func, M_B=None, grad_norms=None, M_O=None, regularization_factor=0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iters=100000, tol=0.1):
    convergence = False
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
        gradient = gradient_func(B, X, O, M_B, M_O, regularization_factor=regularization_factor)

        #For measuring recording gradient norm over time
        if grad_norms != None and i % 100 == 0:
            grad_norms.append(np.linalg.norm(gradient))
        
        # Check for convergence
        if np.linalg.norm(gradient) < tol:
            convergence = True
            # print(f"Convergence reached at iteration {i}")
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
    
    return (O, convergence)


#Predict O function that returns useful information for testing/tuning 
def predict_O_testing(B, X, gradient_func, C=None, grad_norms=None, regularization_factor=0, tol=0.1):
    dim = X.shape[1]
    O = samp.initialize_O(dim)
    return adam_optimizer_testing(B, X, O, gradient_func, M_B=C, grad_norms=grad_norms, regularization_factor=regularization_factor, tol=tol)

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



def predict_B_linear(X, O, C):
    return C * (X @ O @ np.transpose(X))

def predict_B_ReLU(X, O, C):
    return np.maximum(0, X @ O @ np.transpose(X))

def predict_B_tanh(X, O, C):
    return C * np.tanh(X @ O @ np.transpose(X))

def predict_B_rectified_tanh(X, O, C):
    return C * np.maximum(0, np.tanh(X @ O @ np.transpose(X)))

def predict_B_rectified_ztanh(X, O, C):
    mask = ((X @ O @ np.transpose(X)) > 0).astype(int)
    return C * mask * (np.tanh((X @ O @ np.transpose(X)) ** 2))

def predict_B_rectified_itanh(X, O, C):
    mask = ((X @ O @ np.transpose(X)) > 0).astype(int)
    return C * mask * np.tanh(np.abs(X @ O @ np.transpose(X)) ** 0.5)
    #(np.abs(np.tanh((X @ O @ np.transpose(X)))) ** 0.5)
