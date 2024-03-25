from MatrixClass import Matrix
from ActivationFunctions import Activation_ReLu,SoftmaxActivation,Sigmoid
from Layer import FullyConnectedLayer
from LossFunctions import MeanSquaredError
import numpy as np

relu = Activation_ReLu()
inputs = Matrix([[-2.191325326164627],
                [0.14459317022450124],
                [-1.2054476944656431]])
print("Inputs:")
inputs.Print()

print("Outputs:")
output = relu.forward(inputs)
output.Print()


expected_output = Matrix([[0, 0.14459317022450124, 0]])


activation = Sigmoid()
inputs = Matrix([[1.3316355230253767],
                 [-0.023377581119452318]])

output_data_processed = activation.forward(inputs)
print("             ")
print(" Activation output  ")
output_data_processed.Print()

softmax_activation = SoftmaxActivation()
inputs  = Matrix([[0.7911110407860272],
                  [0.4941558708745959]])
output_data_softmax = softmax_activation.forward(inputs)
print("Probabilities: ")
output_data_softmax.Print()

