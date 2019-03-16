import numpy as np
import math
import time
import matplotlib.pyplot as plt
import itertools as it

#https://medium.com/@tm2761/regularization-hyperparameter-tuning-in-a-neural-network-f77c18c36cd3

def sigmoid_derivative(x):
	return x * (1 - x)
def sigmoid(x):
	f_x = 1 / (1 + np.exp(-x))
	return(f_x)
def relu(x, derivative=False):
    if (derivative == True):
        for i in range(0, len(x)):
            for k in range(len(x[i])):
                if x[i][k] > 0:
                    x[i][k] = 1
                else:
                    x[i][k] = 0
        return x
    for i in range(0, len(x)):
        for k in range(0, len(x[i])):
            if x[i][k] > 0:
                pass
                x[i][k] = 0
    return x
class NNet():
	def __init__(self, input_size, hidden_size, output_size, lamda, alpha):
		self.weights1 = np.random.randn(input_size, hidden_size)
		self.weights2 = np.random.randn(hidden_size, output_size)
		self.input_size = input_size #layer size in input
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.lam = lamda #regulization factor
		self.latent1 = np.zeros(hidden_size)
		self.latent2 = np.zeros(output_size)
		self.lr = alpha #learning rate
		self.hidden_layer = np.zeros(hidden_size)
		self.output_layer = np.zeros(output_size)
		self.b1 = np.random.randn(hidden_size) #bias
		self.b2 = np.random.randn(output_size)
	def cost_prime(self, layer):
		dJ = np.multiply(layer, 1-layer)
		return dJ
	def activation(self, input_vector, weights, bias):
		z = input_vector.dot(weights) + bias
		a = sigmoid(z)
		return(z, a)
	def compute_grad(self):
		error_vector = -(self.true_layer - self.output_layer)
		del_output = np.multiply(error_vector, self.cost_prime(self.output_layer))
		del_hidden = np.multiply(del_output.dot(self.weights2.T), self.cost_prime(self.hidden_layer))
		dJ_dweights2 = self.hidden_layer.T.dot(del_output)
		del_b2 = np.sum(del_output, axis = 0)
		del_b2 = del_b2 / self.input_layer.shape[0]
		dJ_dweights1 = self.input_layer.T.dot(del_hidden)
		del_b1 = np.sum(del_hidden, axis = 0)
		del_b1 = del_b1 / self.input_layer.shape[0]
		return(del_b1, del_b2, dJ_dweights1, dJ_dweights2)
	def forward(self):
		self.latent1, self.hidden_layer = self.activation(self.input_layer, self.weights1, self.b1)
		self.latent2, self.output_layer = self.activation(self.hidden_layer, self.weights2, self.b2)
	def train(self, input_vector, true_vector, iterations):
		self.input_layer = input_vector
		self.true_layer = true_vector
		for i in range(0, iterations):
			self.forward()
			del_b1, del_b2, dJ_dweights1, dJ_dweights2 =s self.compute_grad()
			dJ_dweights2 = dJ_dweights2 / input_vector.shape[0] + self.lam * self.weights2
			del_b2 = del_b2 / input_vector.shape[0]
			dJ_dweights1 = dJ_dweights1 / input_vector.shape[0] + self.lam * self.weights1
			del_b1 = del_b1 / input_vector.shape[0]
			self.weights2 = self.weights2 - self.lr * dJ_dweights2
			self.b2 = self.b2 - self.lr * del_b2
			self.weights1 = self.weights1 - self.lr * dJ_dweights1
			self.b1 = self.b1 - self.lr * del_b1
	def predict(self, input_vector):
		self.input_layer = input_vector
		self.forward()
		return(self.output_layer)

# class NNet_old(object):
# 	def __init__(self):
# 		# Generate random numbers
# 		np.random.seed(10)
#
# 		# Assign random weights to a 3 x 1 matrix,
# 		self.synaptic_weights = 2 * np.random.random((2, 1)) - 1
#
# 	# The Sigmoid function
# 	def __sigmoid(self, x):
# 		return 1 / (1 + np.exp(-x))
#
# 	# The derivative of the Sigmoid function.
# 	# This is the gradient of the Sigmoid curve.
# 	def __sigmoid_derivative(self, x):
# 		return x * (1 - x)
#
# 	# Train the neural network and adjust the weights each time.
# 	def train(self, inputs, outputs, training_iterations):
# 		for iteration in range(training_iterations):
#
# 			# Pass the training set through the network.
# 			output = self.learn(inputs)
#
# 			# Calculate the error
# 			error = outputs - output
#
# 			# Adjust the weights by a factor
# 			factor = np.dot(inputs.T, error * self.__sigmoid_derivative(output))
# 			self.synaptic_weights += factor
#
# 	# The neural network thinks.
# 	def learn(self, inputs):
# 		return self.__sigmoid(np.dot(inputs, self.synaptic_weights))
# def sigmoid(x):
#     return 1.0/(1+ np.exp(-x))
# def sigmoid_derivative(x):
#     return x * (1.0 - x)
# class autoencoder_old:
# 	def __init__(self, X, layer_size):
# 		self.input = X
# 		self.weights_encode = np.random.rand(8,3)
# 		self.weights_decode = np.random.rand(3,8)
# 		self.output = np.zeros((8,8))
# 	def feedforward(self):
# 		self.layer_encode = sigmoid(np.dot(self.input, self.weights_encode))
# 		self.layer_decode = sigmoid(np.dot(self.layer_encode, self.weights_decode))
# 	def backprop(self):
# 		self.weights_encode +=
#
# ae = autoencoder(inputs, 3)
# print(ae.layer_encode)
#
# class NNet_old2:
# 	def __init__(self, X, y, layer_size, learning_rate):
# 		self.input      = X
# 		self.weights1   = np.random.rand(self.input.shape[1],layer_size)
# 		self.weights2   = np.random.rand(8,3)
# 		self.y          = y
# 		self.output     = np.zeros(self.input.shape)
# 	def feedforward(self):
# 		self.layer1 = sigmoid(np.dot(self.input, self.weights1)) #bias
# 		self.output = sigmoid(np.dot(self.layer1, self.weights2))
# 		h = self.output
# 		return h
#
# 	def cost(self, X, y):
# 		self.h = self.forwardPropagate(X)
# 		J = 0.5 * sum((y - self.output) ** 2) / X.shape[0] + ( 0.1 /2)*(np.sum(self.weights1**2)+np.sum(self.weights2**2))
# 		return J
# 	def backprop(self):
# 		# application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
# 		d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
# 		d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))
# 		# update the weights with the derivative (slope) of the loss function
# 		self.weights1 += d_weights1 * learning_rate #multply by learning rate
# 		self.weights2 += d_weights2 * learning_rate #multply by learning rate
