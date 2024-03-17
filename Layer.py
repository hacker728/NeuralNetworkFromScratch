from MatrixClass import Matrix
import numpy as np
from ActivationFunctions import Activation_ReLu
class FullyConnectedLayer():
    def __init__(self,input_neurons,next_layer_neurons):
        self.weights = Matrix(np.random.randn(next_layer_neurons,input_neurons))
        # Column vector for bias converted to a Matrix
        self.bias = Matrix(np.random.randn(next_layer_neurons,1))
    def forward(self,inputs: Matrix):
        self.inputs = inputs
        return self.weights.LeftMultiply(inputs).Add(self.bias)
    def backprop(self,Outputgradient :Matrix,learning_rate: float):
        # Output gradient is the derivative of the error with respect to the biases.
        # Learning rate is the fraction of the gradients by which we nudge the hyperparameters.
        ErrorByWeightsGradient = Outputgradient.LeftMultiply(self.inputs.Transpose())
        # This is the derivative of the error with respect to weights.
        self.weights = self.weights.Subtract(ErrorByWeightsGradient.ScalarMultiply(learning_rate))
        # This is updating the bias with derivative of the error with respect to bias(which is the same as the Outputgradient).
        self.bias = self.bias.Subtract(Outputgradient.ScalarMultiply(learning_rate))
        # This is the derivative of error with respect to the inputs.
        return self.weights.Transpose().LeftMultiply(Outputgradient)

# layer1 = FullyConnectedLayer(3,2)
# layer1.weights.Print()
# print("           ")
# layer1.bias.Print()
# print("           ")
# inputs = Matrix( [  [0.1], [0.2], [0.3] ]  )
# layeroutput = layer1.forward(inputs)
# activation1 = Activation_ReLu()
# activation1.forward(layeroutput)
# print("                 ")
# activation1.output.Print()
# # let our Output gradient be [ [0.1],[0.1] ]
# OutputGradient = Matrix([ [0.1],[0.1] ])
# print("           ")
# layer1.backprop(OutputGradient,0.01)
# layer1.weights.Print()
# print("           ")
# layer1.bias.Print()


