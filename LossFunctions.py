from MatrixClass import Matrix
import numpy as np

class MeanSquaredError:
    def __init__(self):
        pass
    def calculate(self, predicted: Matrix, actual: Matrix) -> float:
        # Ensure the dimensions match
        if predicted.rows != actual.rows or predicted.cols != actual.cols:
            raise ValueError("Dimensions of predicted and actual matrices do not match.")
        # Calculate squared differences element-wise
        squared_diff = predicted.Subtract(actual).apply_function(lambda x: x ** 2)
        # Sum of squared differences
        sum_squared_diff = np.sum(squared_diff.ToNumpyArray())
        # Mean Squared Error
        mse = sum_squared_diff / (predicted.rows * predicted.cols)
        return mse

