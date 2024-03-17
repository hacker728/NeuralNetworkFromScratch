import numpy as np
from MatrixClass import Matrix
#from Neuron import  Neuron

class Activation_ReLu:
    def __init__(self):
        pass

    def ReLu(self,z):
        return max(z,0)
    def forward(self,inputs: Matrix):
        # A layer output will be in the form (num_of_neurons,1)
        self.inputs = inputs
        numpyarray = self.inputs.Matrix
        self.output = Matrix(np.maximum(0, numpyarray))
        return self.output
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.numpyarray <= 0] = 0

class Sigmoid:
    def __init__(self):
        pass
    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    def forward(self,input:Matrix):
        sigmoid_output = input.apply_function(self.sigmoid)
        return sigmoid_output

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def backward(self, dvalues: Matrix):
        sigmoid_derivative_output = dvalues.apply_function(lambda x: self.sigmoid_derivative(x))
        return sigmoid_derivative_output
class SoftmaxActivation:
    def __init__(self):
        pass

    def forward(self, inputs: Matrix) -> Matrix:
        # Calculate exponentials of inputs element-wise
        exp_inputs = inputs.apply_function(lambda x: np.exp(x))

        # Calculate the sum of exponentials for each row
        sum_exp = exp_inputs.SumAlongRows()

        # Divide each element by the sum of exponentials for normalization
        probabilitites = exp_inputs
        for i in range(probabilitites.rows):
            row = probabilitites.Matrix[i]
            for j in range(probabilitites.cols):
                value = probabilitites.Matrix[i][j]
                probabilitites.Matrix[i][j] = value / sum_exp[i]

        return probabilitites








