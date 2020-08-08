import numpy as np

def sigmoid(Z):
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    
    A = np.maximum(0,Z)
    cache = Z 
    
    return A, cache

#initializing parameters 
def initialize_parameters_n(layer_dims):
    l = len(layer_dims)
    parameters = {}
    for i in range(1, l):
        parameters["W" + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * 0.01
        parameters["b" + str(i)] = np.zeros((layer_dims[i], 1))
    
    return parameters

#linear forward
def linear_forward(A, W, b):
    
    Z = np.dot(W, A) + b 
    
    cache = (A, W, b)
    
    return Z, cache

#activation
def activation_forward(A_prev, W, b, activation):
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache
    
#propagation
def forward_propagation_n(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2                  
    
    for l in range(1, L):
        A_prev = A 
        A, cache = activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)
    
    AL, cache = activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)
             
    return AL, caches
        
#compute cost      
def compute_cost(AL, Y):
    m = AL.shape[1]
    AL = np.asarray(AL, dtype='float64')
    Y = np.asarray(Y, dtype='float64')
                   
    cost = (1/m) * np.sum(np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1 - AL), 1 - Y))
    cost = np.squeeze(cost)
    return cost

#relu back prop
def relu_backward(dA, cache):
    
    Z = cache
    dZ = np.array(dA, copy=True) 
    
    dZ[Z <= 0] = 0
        
    return dZ
#sigmoid back prop
def sigmoid_backward(dA, cache): 
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    return dZ

#linear backward
def linear_backward(dZ, cache):
    
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m) * (np.dot(dZ, A_prev.T))
    db = (1/m) * (np.sum(dZ, axis = 1, keepdims = True))
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db


def activation_backward(dA, cache, activation):
    
    linear_cache, activation_cache = cache
    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    return dA_prev, dW, db
    
    
def backward_propagation_n(AL, Y, caches):
    grads = {}
    L = len(caches) 
    m = AL.shape[1]
    AL = np.asarray(AL, dtype='float64')
    Y = np.asarray(Y, dtype='float64') 
    
 
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
 
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = activation_backward(dAL, current_cache, "sigmoid")
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = activation_backward(grads["dA" + str(l + 1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 
    
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - (learning_rate* grads["dW" + str(l+1)])
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - (learning_rate* grads["db" + str(l+1)])
    
    return parameters