import numpy as np
from MatrixClass import Matrix
from ActivationFunctions import Activation_ReLu, SoftmaxActivation
from LossFunctions import MeanSquaredError
from Layer import FullyConnectedLayer
import pandas as pd
import matplotlib.pyplot as plt


# Load the dataset
train_data = pd.read_csv('emnist-balanced-train.csv',header = None ,  index_col=0)
test_data = pd.read_csv('emnist-balanced-test.csv',header = None ,  index_col=0)

#training_characters
y_train = np.array(train_data.iloc[:,0].values,dtype= np.float64)
x_train = np.array(train_data.iloc[:,1:].values,dtype= np.float64)

#testing_labels
y_test = np.array(test_data.iloc[:,0].values,dtype= np.float64)
x_test = np.array(test_data.iloc[:,1:].values,dtype= np.float64)

# Normalize
x_train  = x_train /255
y_train = y_train /255

#Rotate EMNIST data
rotated_train = []
for x in range(len(x_train)):
    resized = np.resize(x_train[x][1:],(28,28))
    flipped = np.fliplr(resized)
    rotated = np.rot90(flipped)
    rotated_train.append(rotated)

rotated_test = []
for x in range(len(x_test)):
    resized = np.resize(x_test[x][1:],(28,28))
    flipped = np.fliplr(resized)
    rotated = np.rot90(flipped)
    rotated_test.append(rotated)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize layers
        self.fc1 = FullyConnectedLayer(input_size, hidden_size)
        self.relu = Activation_ReLu()
        self.fc2 = FullyConnectedLayer(hidden_size, output_size)
        self.relu2 =  Activation_ReLu()
        self.softmax = SoftmaxActivation()

    def forward(self, inputs):
        # Forward pass
        hidden_output = self.fc1.forward(inputs)
        relu_output = self.relu.forward(hidden_output)
        output = self.fc2.forward(relu_output)
        output = self.relu2.forward(output)
        output = self.softmax.forward(output)
        return output

    def train(self, inputs, targets):
        # Forward pass
        output = self.forward(inputs)

        # Compute loss
        loss = MeanSquaredError().calculate(output, targets)

        # Backpropagation
        output_gradient = output.Subtract(targets)
        self.fc2.backprop(output_gradient, self.learning_rate)
        relu_gradient = self.fc2.weights.Transpose().LeftMultiply(output_gradient)
        self.relu.backward(relu_gradient)
        self.fc1.backprop(relu_gradient, self.learning_rate)

        return loss

    def predict(self, inputs):
        # Forward pass for prediction
        output = self.forward(inputs)
        return output


# Define network parameters
input_size = 784  # 28x28 pixels
hidden_size = 128
output_size = 47  # Number of classes (26 uppercase letters and 11 lowercase letters and 10 digits.)
learning_rate = 0.01
batch_size = 64
num_epochs = 5

# Create the neural network
nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)

for epoch in range(num_epochs):
    total_loss = 0.0
    for i in range(0, len(rotated_train), batch_size):
        batch_inputs = []
        for image in rotated_train[i:i + batch_size]:
            flattened_image = image.flatten()  # Flatten the image into a 1D array
            matrix_input = Matrix([flattened_image])  # Convert the flattened array into a Matrix object
            transposed_input = matrix_input.Transpose()  # Transpose the matrix to match the input shape expected by the neural network
            batch_inputs.append(transposed_input)  # Add the transposed input to the batch_inputs list

        batch_targets = []
        for label in y_train[i:i + batch_size]:
            label_matrix = Matrix([[label]])  # Create a Matrix object for the label
            batch_targets.append(label_matrix)  # Add the label matrix to the batch_targets list

        batch_loss = 0.0
        for j in range(len(batch_inputs)):
            loss = nn.train(batch_inputs[j], batch_targets[j])
            batch_loss += loss
        total_loss += batch_loss / len(batch_inputs)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / (len(rotated_train) / batch_size)}")



