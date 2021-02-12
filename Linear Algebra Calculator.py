# Matrix Calculator, this will just be useful in my Linear Algebra Studies
import math
import copy
import random
'''
Valid Operations:
Addition - Done
Matrix Scalar Multiplication - Done
Subtraction - Done
MatMul - Done
RREF - Done
RREF With Augmented Matrix - Done
Recursive Determinants
'''
class Matrix():
  def __init__(self, height, width):
    self.height = height
    self.width = width
    self.rows = []
    self.columns = []
    for i in range(self.height):
      row = [0] * self.width
      self.rows += [row]
    for i in range(self.width):
      column = [0] * self.height
      self.columns += [column]
  def transpose(self, matrix):
    '''
    Matrix: This is the matrix to be transposed, not in place.
    '''
    height = len(matrix)
    width = len(matrix[0])
    new_matrix = []
    # Set up new transposed matrix array
    for i in range(width):
      column = [0] * height
      new_matrix += [column]
    # Start filling in values
    for i in range(height):
      for j in range(width):
        new_matrix[j][i] = matrix[i][j]
    return new_matrix
  def __str__(self, height = False):
    '''
    Prints a Representation of a matrix
    height: Whether to print the height representation(columns) or the rows
    '''
    if not height:
      for i in self.rows:
        print(i)
    else:
      for i in range(self.width):
        vals = []
        for column in self.columns:
          vals += [column[i]]
        print(vals)
    return f"{self.height} x {self.width} Matrix"
  def update_item(self, row, column, value):
    self.columns[column][row] = value
    self.rows[row][column] = value
  def matrify(self, values):
    '''
    1D array of values to quickly fill in the matrix
    Values should be aligned with rows
    '''
    assert len(values) == self.width * self.height
    for row in range(self.height):
      for column in range(self.width):
        self.rows[row][column] = values[row * self.width+ column]
    self.columns = self.transpose(copy.deepcopy(self.rows))
  def same_shape(self, B):
    '''
    Returns if two matrices have the same shape, used for testing
    '''
    return self.height == B.height and self.width == B.width
class VectorCalculator():
    '''
    This isnt an extensive class, but it performs basic vector operations to process matrices
    Vectors in this class are just 1D lists
    '''
    def __init__(self):
        pass # Again, it's a Calculator
    def add(self, A, B):
        '''
        This adds two vectors of same length
        '''
        assert len(A) == len(B)
        new_vec = []
        for i in range(len(A)):
            new_vec += [A[i] + B[i]]
        return new_vec
    def scal_mult(self, A, scalar):
        new_vec = []
        for i in A:
            new_vec += [i * scalar]
        return new_vec
    def sub(self, A, B):
        # No need for a check, __add__ will do this
        return self.add(A, self.scal_mult(B, -1))
    def dot_prod(self, A, B):
        assert len(A) == len(B)
        dot_prod = 0.0
        for i in range(len(A)):
            dot_prod += A[i] * B[i]
        return dot_prod
class Calculator():
    def __init__(self):
        self.VectorCalc = VectorCalculator()
    def mat_add(self, A, B):
        '''
        Adds Matrix B to matrix A, creating a new Matrix instance
        Matrix B should be another matrix that matches shape to matrix A
        '''
        # Create new Matrix instance
        assert A.same_shape(B)
        matrix = Matrix(A.height, A.width)
        for row in range(A.height):
          for column in range(A.width):
            matrix.update_item(row, column, A.rows[row][column] + B.rows[row][column])
        return matrix
    def mat_scal_mult(self, A, scalar):
        '''
        Multiplies a Scalar to a Matrix
        A: Matrix of any size or shape
        integer: just a scalar multiple
        '''
        matrix = Matrix(A.height, A.width)
        for i in range(A.height):
            for j in range(A.width):
                matrix.update_item(i, j, scalar * A.rows[i][j])
        return matrix
    def mat_sub(self, A, B):
        '''
        A: Matrix of any size
        B: Matrix of the same shape
        A - B is this operation
        '''
        return self.mat_add(A, self.mat_scal_mult(B, -1.0))
    def mat_mul(self, A, B):
        # Check that the matrices are compatible
        assert A.width == B.height # This will ensure that the matrices are matmulable
        matrix = Matrix(A.height, B.width)
        for row_idx in range(len(A.rows)):
            row = A.rows[row_idx]
            for col_idx in range(len(B.columns)):
                col = B.columns[col_idx]
                matrix.update_item(row_idx, col_idx, self.VectorCalc.dot_prod(row, col))
        return matrix
    def simplify_row(self, row):
        '''
        Reduces the first non zero element down to one, and reduces everything else the same, returning the position of the first 1
        '''
        idx = 0
        val = 0
        for i in range(len(row)):
            if row[i] != 0:
                idx = i
                val = row[i]
                break
        if val == 0:
            return None, None
        return self.VectorCalc.scal_mult(row, 1 / val), idx
    def RREF(self, A):
        '''
        Returns a Row Reduced Form of the A matrix, computes rows first correctly, and then just creates a matrix for simplicity
        '''
        new_mat = A
        for idx in range(len(A.rows)):
            new_mat = self.RREF_sort(new_mat)
            reduced_row, col_idx = self.simplify_row(new_mat.rows[idx])
            if col_idx == None:
                break
            new_mat.rows[idx] = reduced_row
            for row_idx in range(len(new_mat.rows)):
                if row_idx != idx:
                    other_row = new_mat.rows[row_idx]
                    RR = self.VectorCalc.sub(other_row, self.VectorCalc.scal_mult(reduced_row, other_row[col_idx]))
                    new_mat.rows[row_idx] = RR
        return self.RREF_sort(new_mat)
    def RREF_sort(self, A):
        '''
        Rearranges Rows(Columns dont matter RN) and creates a new matrix with it
        '''
        new_rows = []
        indices_selected = {}
        for i in range(len(A.rows)):
            indices_selected[i] = False
        for i in range(A.width):
            for row_idx in range(len(A.rows)):
                row = A.rows[row_idx]
                if row[i] != 0 and not indices_selected[row_idx]:
                    new_rows += [row]
                    indices_selected[row_idx] = True
        matrix = Matrix(A.height, A.width)
        # flatten the row_list
        flat = []
        for row in new_rows:
            for item in row:
                flat += [item]
        # pad up to necessary size
        while len(flat) != A.height * A.width:
            flat += [0]
        matrix.matrify(flat)
        return matrix
    def aug_RREF(self, A, aug_line):
        '''
        Performs an augmented Row Reduction with the Gauss Jordan Method
        A: matrix of any size
        aug_line: int, tells the program what index do augmented numbers begin(0 Based)
        '''
        assert aug_line > -1 and aug_line < A.width
        cur_mat = A
        for i in range(len(A.rows)):
            cur_mat, indices = self.aug_RREFSort(cur_mat, aug_line)
            if indices[i] == -1:
                return cur_mat
            cur_row = cur_mat.rows[i]
            if cur_row[indices[i]] < 0.01:
                continue # This corrects for floating point imprecision
            cur_row = self.VectorCalc.scal_mult(cur_row, 1 / cur_row[indices[i]])
            cur_mat.rows[i] = cur_row
            for row_idx in range(len(A.rows)):
                if cur_mat.rows[row_idx][indices[i]] != 0 and row_idx != i:
                    cur_mat.rows[row_idx] = self.VectorCalc.sub(cur_mat.rows[row_idx], self.VectorCalc.scal_mult(cur_row, cur_mat.rows[row_idx][indices[i]]))
        return cur_mat
    def aug_RREFSort(self, A, aug_line):
        '''
        Fundamentally the same operation as RREFSort, but ignores everything past the augmented line
        '''
        assert aug_line > -1 and aug_line < A.width
        new_rows = []
        reduced = {}
        indices = {} # Stores where each row has a non zero entry 
        for i in range(len(A.rows)):
            reduced[i] = False
        for i in range(A.width):
            for row_idx in range(len(A.rows)):
                if A.rows[row_idx][i] != 0 and not reduced[row_idx]:
                    new_rows += [A.rows[row_idx]]
                    reduced[row_idx] = True
                    indices[row_idx] = i
        for i in reduced:
            if not reduced[i]: # This is an impossible matrix, but RREF doesn't really change have any difference
                fake = False
                for item in A.rows[i]:
                    if item != 0:
                        fake = True
                if fake:
                    print('It seems that this matrix has no real solution')
                new_rows += [A.rows[i]]
                indices[row_idx] = -1
        new_mat = Matrix(A.height, A.width)
        # Flatten new_rows
        flat = []
        for row in new_rows:
            for entry in row:
                flat += [entry]
        new_mat.matrify(flat)
        return new_mat, indices
    def det(self, A):
        '''
        Recursively computes the determinant of a given matrix
        '''
        # Determinants can only be computed on Square matrices, so lets check for that
        assert A.height == A.width, "Determinants only work on Square Matrices"
        if A.height == 1:
            return A.rows[0][0] # The determinant at this point is just the singular item
        else:
            det = 0.0
            i = 0 # 0 based, but in computing the cofactor expansion, I will act as if this is 1
            for j in range(len(A.columns[0])):
                new_mat = Matrix(A.height - 1, A.width - 1)
                # compute the cofactor expansion
                row_idx = 0
                col_idx = 0
                for k in range(len(A.rows[0])):
                    for l in range(len(A.columns[0])):
                        if k != i and l != j:
                            new_mat.update_item(row_idx, col_idx, A.rows[k][l])
                            col_idx += 1
                            if col_idx == new_mat.width:
                                col_idx = 0
                                row_idx += 1
                det += A.rows[i][j] * ((-1) ** (i + 2 + j)) * self.det(new_mat)
            return det
                        
        

class Fraction():
    def __init__(self):
        self.top = None
        self.bottom = None
    def __str__(self):
        return f"{self.top} / {self.bottom}"
    def convert_decimal_to_frac(self, decimal):
        '''
        The Matrix Calculator outputs decimals, we will reduce down fractions to get a fractional representation
        decimal is essentially the output of one square in the matrix
        '''
        self.bottom = ''
        while int(decimal) != decimal:
            decimal *= 10
            self.bottom += '9'
        if self.bottom == '':
            self.top = int(decimal)
            self.bottom = 1
            return
        self.top = int(decimal)
        if self.top == 0:
            self.bottom = 1
            return
        self.bottom = int(self.bottom)
        i = 2
        while i <= abs(self.top):
            if self.top % i == 0 and self.bottom % i == 0:
                self.top = self.top // i
                self.bottom = self.bottom // i
                i = 2
            else:
                i += 1
        return self

matrix = Matrix(3, 3) # Simple 3x3 Matrix
matrix.matrify([1, 3, 2, -3, -1, -3, 2, 3, 1])

# Create Calculator
calculator = Calculator()
print(calculator.det(matrix))

