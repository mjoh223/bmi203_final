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

########################################################################################
########################################################################################
def sigmoid(x):
    return 1.0/(1+ np.exp(-x))
def sigmoid_derivative(x):
    return x * (1.0 - x)
class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],10)
        self.weights2   = np.random.rand(10,1)
        self.y          = y
        self.output     = np.zeros(self.y.shape)
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1 #multply by learning rate
        self.weights2 += d_weights2 #multply by learning rate

inputs = np.array([[0, 1, 0], [1, 0, 1]]) #row is the observations columns are the features
outputs = np.array([[1, 0]]).T
nn = NeuralNetwork(inputs, outputs)
for i in range(150):
    nn.feedforward()
    nn.backprop()
    print(nn.output)


if __name__ == "__main__":
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

	inputs = np.array([[0, 1, 0, 1], [1, 0, 1, 1]]) #row is the observations columns are the features
	outputs = np.array([[1, 0]]).T

	# Train the neural network

		#Initialize
	neural_network = NeuralNet()
	neural_network.train(inputs, outputs, 10000)

	# Test the neural network with a test example.
	#print(neural_network.learn(np.array([1, 0, 1])))
