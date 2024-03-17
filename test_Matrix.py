import unittest
from MatrixClass import Matrix

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

if __name__ == '__main__':
    unittest.main()