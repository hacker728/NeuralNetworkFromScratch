from MatrixClass import Matrix
from ActivationFunctions import Activation_ReLu,SoftmaxActivation,Sigmoid
from Layer import FullyConnectedLayer
from LossFunctions import MeanSquaredError
import numpy as np

input_neurons = 2
hidden_neurons = 3
output_neurons = 2

hidden_layer = FullyConnectedLayer(input_neurons, hidden_neurons)
output_layer = FullyConnectedLayer(hidden_neurons, output_neurons)

input_data = Matrix([[0.1], [0.2]])
hidden_output = hidden_layer.forward(input_data)

activation1 = Activation_ReLu()
activation_output = activation1.forward(hidden_output)


# Forward pass through output layer
output_data = output_layer.forward(activation_output)
output_data.Print()

activation2 = Sigmoid()
output_data_processed = activation2.forward(output_data)
print("             ")
output_data_processed.Print()

softmax_activation = SoftmaxActivation()
output_data_softmax = softmax_activation.forward(output_data_processed)
print("             ")
output_data_softmax.Print()

output_data_processed = Matrix([[0.5],[0.5]])

target_output = Matrix([[0.7], [0.3]])
mse_loss = MeanSquaredError()
loss = mse_loss.calculate(output_data_softmax, target_output)
print("Mean Squared Error Loss:", loss)

# Backpropagation

# Calculate output gradient
output_gradient = output_data_sigmoid.subtract(target_output)
# Backpropagate through output layer
output_layer.backprop(output_gradient, learning_rate=0.01)
# Backpropagate through hidden layer
hidden_gradient = output_layer.weights.Transpose().LeftMultiply(output_gradient)
hidden_layer.backprop(hidden_gradient, learning_rate=0.01)