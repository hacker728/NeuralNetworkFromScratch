import unittest
from MatrixClass import Matrix
import math

class test_matrix(unittest.TestCase):
    def setUp(self):
        self.MatrixA = Matrix([[1, 2, 3], [4, 5, 6],[7, 8, 9]])
        self.MatrixB = Matrix([[7,8],[9,10]])
        self.MatrixC = Matrix([[2,4,6],[9,3,4],[10,7,1]])
    def test_addition(self):
        result = self.MatrixA.Add(self.MatrixC)
        expected_result = Matrix([[3,6,9],[13,8,10],[17,15,10]])
        self.assertEqual(result,expected_result,"Matrix addition failed")
    def test_subtraction(self):
        result = self.MatrixC.Subtract(self.MatrixA)
        expected_result = Matrix([[1,2,3],[5,-2,-2],[3,-1,-8]])
        self.assertEqual(result, expected_result, "Matrix subtractiom failed")
    def test_left_multiplication(self):
        result = self.MatrixC.LeftMultiply(self.MatrixA)
        expected_result = Matrix([[50,31,17],[113,71,50],[176,115,83]])
        self.assertTrue(result,expected_result)
    def test_transposition(self):
        result = self.MatrixA.Transpose()
        expected_result = Matrix([[1,4,7],[2,5,8],[3,6,9]])
        self.assertEqual(result,expected_result,"Matrix transposition failed")
    def test_equality(self):
        copy = Matrix([[1, 2, 3], [4, 5, 6],[7, 8, 9]])
        self.assertEqual(self.MatrixA, copy, "Matrix Equality failed")
    def test_scalar(self):
        result = self.MatrixA.ScalarMultiply(2)
        expected_result = Matrix([[2, 4, 6], [8, 10, 12],[14, 16, 18]])
        self.assertEqual(result,expected_result,"Matrix scalar multiplication failed")

    def test_exponential(self):
        result = self.MatrixA.exponenentiate()
        expected_result = Matrix([[math.e**(1), math.e**(2), math.e**(3)],
                                  [math.e**(4), math.e**(5), math.e**(6)],
                                  [math.e**(7), math.e**(8), math.e**(9)]])
        self.assertEqual(result, expected_result, "Matrix exponentiation failed")

    def test_sum_along_rows(self):
        result = self.MatrixA.SumAlongRows()
        expected_result = [6, 15, 24]
        self.assertEqual(result, expected_result, "Sum along rows failed")

    def test_sum_along_cols(self):
        result = self.MatrixA.SumAlongCols()
        expected_result = [12, 15, 18]
        self.assertEqual(result, expected_result, "Sum along columns failed")

    def test_apply_function(self):
        def square(x):
            return x ** 2

        result = self.MatrixA.apply_function(square)
        expected_result = Matrix([[1, 4, 9], [16, 25, 36], [49, 64, 81]])
        self.assertEqual(result, expected_result, "Applying function to matrix elements failed")


if __name__ == '__main__':
    unittest.main()