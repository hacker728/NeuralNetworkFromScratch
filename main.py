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
print(" Hidden Layer weights:")
hidden_layer.weights.Print()
print("                 ")
print(" Hidden Layer bias:")
hidden_layer.bias.Print()
print("                 ")
print(" Output Layer weights:")
output_layer.weights.Print()
print("                 ")
print(" Output Layer bias:")
output_layer.bias.Print()
print("             ")

input_data = Matrix([[0.1], [0.2]])
hidden_output = hidden_layer.forward(input_data)
print("Weighted inputs (First pass) ")
hidden_output.Print()

activation1 = Activation_ReLu()
activation_output = activation1.forward(hidden_output)
print("Activation outputs (First pass) ")
activation_output.Print()

# Forward pass through output layer
output_data = output_layer.forward(activation_output)
print("Weighted inputs (Second pass) ")
output_data.Print()

activation2 = Sigmoid()
output_data_processed = activation2.forward(output_data)
print("             ")
print(" Activation output (Second Pass) ")
output_data_processed.Print()

softmax_activation = SoftmaxActivation()
output_data_softmax = softmax_activation.forward(output_data_processed)
print("             ")
print("Probabilities: ")
output_data_softmax.Print()



target_output = Matrix([[0.7], [0.3]])
mse_loss = MeanSquaredError()
loss = mse_loss.calculate(output_data_softmax, target_output)
print("Mean Squared Error Loss:", loss)

# Backpropagation

# Calculate output gradient
output_gradient = output_data_softmax.Subtract(target_output)
# Backpropagate through output layer
output_layer.backprop(output_gradient, learning_rate=0.01)
# Backpropagate through hidden layer
hidden_gradient = output_layer.weights.Transpose().LeftMultiply(output_gradient)
hidden_layer.backprop(hidden_gradient, learning_rate=0.01)

print(" New Layer 1-2 weights:")
hidden_layer.weights.Print()
print("                 ")
print(" New Layer 1-2 biases:")
hidden_layer.bias.Print()
print("                 ")
print(" New Layer 2-3 weights:")
output_layer.weights.Print()
print("                 ")
print(" New Layer 2-3 biases:")
output_layer.bias.Print()

# Trying again to see if Loss reduces

input_data = Matrix([[0.1], [0.2]])
hidden_output = hidden_layer.forward(input_data)

activation1 = Activation_ReLu()
activation_output = activation1.forward(hidden_output)

# Forward pass through output layer
output_data = output_layer.forward(activation_output)


activation2 = Sigmoid()
output_data_processed = activation2.forward(output_data)


softmax_activation = SoftmaxActivation()
output_data_softmax = softmax_activation.forward(output_data_processed)


target_output = Matrix([[0.7], [0.3]])
mse_loss = MeanSquaredError()
loss = mse_loss.calculate(output_data_softmax, target_output)
print("Mean Squared Error Loss:", loss)



