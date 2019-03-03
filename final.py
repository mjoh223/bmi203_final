import csv
import numpy as np
import pandas as pd

class NeuralNet(object):
	def __init__(self):
		# Generate random numbers
		np.random.seed(10)

		# Assign random weights to a 3 x 1 matrix,
		self.synaptic_weights = 2 * np.random.random((2, 1)) - 1

	# The Sigmoid function
	def __sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	# The derivative of the Sigmoid function.
	# This is the gradient of the Sigmoid curve.
	def __sigmoid_derivative(self, x):
		return x * (1 - x)

	# Train the neural network and adjust the weights each time.
	def train(self, inputs, outputs, training_iterations):
		for iteration in range(training_iterations):

			# Pass the training set through the network.
			output = self.learn(inputs)

			# Calculate the error
			error = outputs - output

			# Adjust the weights by a factor
			factor = np.dot(inputs.T, error * self.__sigmoid_derivative(output))
			self.synaptic_weights += factor

	# The neural network thinks.
	def learn(self, inputs):
		return self.__sigmoid(np.dot(inputs, self.synaptic_weights))

if __name__ == "__main__":

	#Initialize
	neural_network = NeuralNet()

	# The training set.
	positives = []
	with open('/Users/matt/OneDrive/UCSF/algorithms/final/rap1-lieb-positives.txt') as csv_file:
	    csv_reader = csv.reader(csv_file, delimiter=',')
	    for row in csv_reader:
	        positives.append(row)

	positives = np.asarray(positives).flatten()
	conversion_dictionary = {
	"A": [0,0],
	"T": [0,1],
	"G": [1,0],
	"C": [1,1]
	}
	seq_list = []
	for seq in positives:
		encoding_list = []
		for nt in seq:
			encoding = conversion_dictionary[nt]
			encoding_list.append(encoding)
		seq_list.append(encoding_list)

	positives_encoded = pd.DataFrame(seq_list)
	outputs_test = np.ones((137,1))
	positives_encoded_array = np.asarray(seq_list)

	inputs = np.array([[0, 1], [1, 0], [1, 0]])
	outputs = np.array([[1, 0]]).T

	# Train the neural network
	neural_network.train(inputs, outputs, 10000)

	# Test the neural network with a test example.
	#print(neural_network.learn(np.array([1, 0, 1])))
