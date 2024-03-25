import numpy as np
import math
class Matrix:
    def __init__(self,Matrix):
        # Checking if all values are numeric.
        for row in Matrix:
            for value in row:
                if not isinstance(value, (int, float)):
                    raise ValueError("Matrix elements must be numeric.")

        rows = len(Matrix) #This is the number of rows.
        cols = len(Matrix[0]) #Cheking the number of columns is consistant for each element of the array otherwise it is not a Matrix.
        valid = True
        for i in range(1, rows):
            if len(Matrix[i]) != cols:
                valid = False
                break
        if (valid == False):
            raise ValueError("Columns not consistant")
        self.Matrix = Matrix  # This will be a Multidimensional Array
        self.rows = rows
        self.cols = cols

    def Print(self):
        for i in range (0,self.rows):
            print(self.Matrix[i])

    def __eq__(self, other):
        #Check if other is also an instance of the Matrix class
        if not isinstance(other, Matrix):
            return False

        # Check if dimensions are the same
        if self.rows != other.rows or self.cols != other.cols:
            return False
        # Compare each element at respective positions
        for i in range(self.rows):
            for j in range(other.cols):
                if(self.Matrix[i][j] != other.Matrix[i][j]):
                    return False
        return True

    def LeftMultiply(self,other):
        if (self.cols != other.rows):
            raise ValueError("Matrix 1 columns must be equal to Matrix 2 rows.")

        result = []
        for _ in range(self.rows):
            inner_list = []
            for _ in range(other.cols):
                inner_list.append(0)
            result.append(inner_list)
        # Creates a multidimensionalarray of dimensions [rows][columns]

        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    result[i][j] += self.Matrix[i][k] * other.Matrix[k][j]
        return Matrix(result)

    def ScalarMultiply(self,factor):
        result_matrix = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(0)
            result_matrix.append(row)


        for i in range(self.rows):
            for j in range(self.cols):
                result_matrix[i][j] = self.Matrix[i][j] * factor
        return Matrix(result_matrix)

    def exponenentiate(self):
        result_matrix = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(0)
            result_matrix.append(row)

        for i in range(self.rows):
            for j in range(self.cols):
                result_matrix[i][j] = math.e ** (self.Matrix[i][j])
        return Matrix(result_matrix)

    def Transpose(self):
        transposed = []
        for i in range(self.cols):
            inner_list = []
            for j in range(self.rows):
                inner_list.append(self.Matrix[j][i])
            transposed.append(inner_list)

        return Matrix(transposed)
    def Add(self,other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions for addition")

        result = []
        for i in range(self.rows):
            new_row = []
            for j in range(self.cols):
                element_sum = self.Matrix[i][j] + other.Matrix[i][j]
                new_row.append(element_sum)
            result.append(new_row)

        return Matrix(result)

    def Subtract(self,other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions for subtraction")

        result = []
        for i in range(self.rows):
            new_row = []
            for j in range(self.cols):
                element_sum = self.Matrix[i][j] - other.Matrix[i][j]
                new_row.append(element_sum)
            result.append(new_row)

        return Matrix(result)

    def ToNumpyArray(self):
        return np.array(self.Matrix)

    def SumAlongRows(self):
        sum = [0]* self.rows
        for i in range(self.rows):
            for j in range(self.cols):
                sum[i] += self.Matrix[i][j]
        return sum

    def SumAlongCols(self):
        sum = [0] * self.cols
        for i in range(self.cols):
            for j in range(self.rows):
                sum[i] += self.Matrix[j][i]
        return sum

    def apply_function(self, func):
        # Apply a function to each element of the matrix
        result = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(func(self.Matrix[i][j]))
            result.append(row)
        return Matrix(result)











