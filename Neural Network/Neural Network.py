import numpy as np
import csv

#This is a 3 layers neural network with 1st layer 784 nodes, 2nd layer 100 node and 3rd layer 10 node.
#Input is 28*28 handwritten number image. Each image is represented as an array.
class NeuralNetwork:
    def __init__(self, input_layer_nodes, hidden_layer_nodes, output_layer_nodes, learning_rate):
        self.inputNodes = input_layer_nodes
        self.hiddenNodes = hidden_layer_nodes
        self.outputNodes = output_layer_nodes
        self.learningRate = learning_rate

        # Weight range : If each node has 100 incoming links, the weights should be in the range from
        # -1/sqrt(100) to +1/sqrt(100) or -+0.1 (From slide)

        self.W_input_hidden_max_range = 1 / np.sqrt(input_layer_nodes)
        self.W_input_hidden_min_range = -1 / np.sqrt(input_layer_nodes)

        self.W_hidden_output_max_range = 1 / np.sqrt(hidden_layer_nodes)
        self.W_hidden_output_min_range = -1 / np.sqrt(hidden_layer_nodes)

        # To generate a random number from range [a,b) b>a : (b - a) * np.random.random(row,column) + a
        self.W_input_hidden = (self.W_input_hidden_max_range - self.W_input_hidden_min_range) * np.random.random(
            (hidden_layer_nodes, input_layer_nodes)) + self.W_input_hidden_min_range

        self.W_hidden_output = (self.W_hidden_output_max_range - self.W_hidden_output_min_range) * np.random.random(
            (output_layer_nodes, hidden_layer_nodes)) + self.W_hidden_output_min_range

    @staticmethod
    def derivatives_sigmoid(x):
        return x * (1 - x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def query(self, i):
        I = i

        # Forward propagate, from input layer to hidden layer
        X_hidden = np.dot(self.W_input_hidden, I)
        self.O_hideen = self.sigmoid(X_hidden)
        # Forward propagate, from hidden layer to output layer
        X_output = np.dot(self.W_hidden_output, self.O_hideen)
        self.O_output = self.sigmoid(X_output)

        return self.O_output

    def train(self, i, target):
        target = np.array(target)
        self.I = i
        self.O_output = self.query(self.I)

        error_output_layer = target - self.O_output
        error_hidden_layer = np.dot(self.W_hidden_output.T, error_output_layer)

        delta_W_hidden_output = (-1 * error_output_layer * self.derivatives_sigmoid(self.O_output)).dot(self.O_hideen.T)
        delta_W_input_hidden = (-1 * error_hidden_layer * self.derivatives_sigmoid(self.O_hideen)).dot(self.I.T)

        self.W_hidden_output = self.W_hidden_output - self.learningRate * delta_W_hidden_output
        self.W_input_hidden = self.W_input_hidden - self.learningRate * delta_W_input_hidden
