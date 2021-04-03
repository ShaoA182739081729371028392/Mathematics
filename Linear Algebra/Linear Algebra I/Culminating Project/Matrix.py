import copy
import math
from Vector import Vector
#from Coordinate import Coordinate
#from Collection import Collection
'''
This Change of Basis Calculator will convert vectors into certain coordinates and back. It will also compute change of basis matrices.

Supported Operations:
- Find a basis
- Convert to coordinates
- Calculate change of basis
Helpers:
- Matrix Multiply - Done
- RREF - Done
- Addition Subtraction, etc. - Done
'''
class Matrix:
  '''
  Holds the necessary functions for a Matrix.
  '''
  def __init__(self, width, height):
    self.width = width
    self.height = height
    # Fill Empty Place Holder
    self.values = [[0 for w in range(width)] for h in range(height)] # Construct a matrix(row x column indexing)
  
  def __add__(self, other):
    if isinstance(other, Matrix):
      return Matrix.add(self, other)
    else:
      # Form Matrix filled with val_other
      W, H = self.get_size()
      matrix = Matrix(W, H)
      for w in range(W):
        for h in range(H):
          matrix.set_value(h, w, other)
      return Matrix.add(self, matrix)
  def __mul__(self, other):
    if isinstance(other, Matrix):
      return Matrix.matmul(self, other)
    else:
      return Matrix.scal_mult(self, other)
  def is_square(self):
    return self.width == self.height
  def fill_identity(self):
    # Fills the Matrix with the identity matrix
    assert self.width == self.height, "Matrix should be square"

    for i in range(self.width):
      for j in range(self.height):
        if i == j:
          self.values[i][i] = 1
        else:
          self.values[i][j] = 0
  def list_fill(self, values):
    '''
    Fills the matrix.
    '''
    self.values = values
  def set_row(self, row_idx, row_val):
    self.values[row_idx] = row_val
  def set_col(self, col_idx, col_val):
    for row_idx in range(self.height):
      self.values[row_idx][col_idx] = col_val[row_idx]
  def get_row(self, row):
    return self.values[row]
  def get_col(self, col):
    values = [row[col] for row in self.values]
    return values
  def get_size(self):
    return (self.width, self.height)
  def get_value(self, row, column):
    return self.values[row][column]
  def set_value(self, row, column, value):
    self.values[row][column] = value
  def __str__(self):
    for row in self.values:
      print(row)
    return f"{self.height} x {self.width} Matrix"
  def set_value(self, row, column, value):
    self.values[row][column] = value
  # Basic Arithmetic
  @classmethod
  def comparison(cls, mat_A, mat_B):
    width, height = mat_A.get_size()
    if mat_B.get_size() != (width, height):
      return False
    for i in range(width):
      for j in range(height):
        if mat_A.get_value(i, j) != mat_B.get_value(i, j):
          return False
    return True
  @classmethod
  def add(cls, mat_A, mat_B):
    size = mat_A.get_size()
    assert size == mat_B.get_size()
    width, height = size
    new_mat = Matrix(size[0], size[1])
    for w in range(width):
      for h in range(height):
        new_mat.set_value(h, w, mat_A.get_value(h, w) + mat_B.get_value(h, w))
    return new_mat
  @classmethod
  def scal_mult(cls, mat_A, scal):
    width, height = mat_A.get_size()
    new_mat = Matrix(width, height)
    for w in range(width):
      for h in range(height):
        new_mat.set_value(h, w, mat_A.get_value(h, w) * scal)
    return new_mat
  @classmethod
  def matmul(cls, mat_A, mat_B):
    A_width, A_height = mat_A.get_size()
    B_width, B_height = mat_B.get_size()
    assert A_width == B_height
    new_mat = Matrix(B_width, A_height)
    for i in range(A_height):
      for j in range(B_width):
        A_row = Vector(mat_A.get_row(i))
        B_col = Vector(mat_B.get_col(j))
        new_mat.set_value(i, j, Vector.dot(A_row, B_col))
    return new_mat
  @classmethod
  def sub(cls, mat_A, mat_B):
    return cls.add(mat_A, cls.scal_mult(mat_B, -1))
  # Elementary Row Operations
  @classmethod
  def matrix_concat_row(cls, mat_A, mat_B):
    '''
    Concatenates Two Matrices:(rowwise, so the height will increase.)
    '''
    width_A, height_A = mat_A.get_size()
    width_B, height_B = mat_B.get_size()
    assert width_A == width_B
    values_A = mat_A.values
    values_B = mat_B.values
    concat_vals = values_A + values_B

    new_mat = Matrix(width_A, height_A + height_B)
    new_mat.list_fill(concat_vals)
    return new_mat
  @classmethod
  def matrix_concat_column(cls, mat_A, mat_B):
    '''
    Concatenates Two Matrices: (columnwise)
    '''
    # Transpose both matrices
    trans_A = cls.transpose(mat_A)
    trans_B = cls.transpose(mat_B)
    # Concat as usual
    concat = cls.matrix_concat_row(trans_A, trans_B)
    # Transpose Back
    return cls.transpose(concat)
  @classmethod
  def swap_two_rows(cls, mat_A, row1_idx, row2_idx):
    '''
    Swaps Row1 and Row2
    '''
    row1 = mat_A.get_row(row1_idx)
    row2 = mat_A.get_row(row2_idx)
    width, height = mat_A.get_size()
    new_mat = Matrix(width, height)
    new_mat.list_fill(mat_A.values)

    new_mat.set_row(row1_idx, row2)
    new_mat.set_row(row2_idx, row1)
    return new_mat
  @classmethod
  def transpose(cls, mat_A):
    width, height = mat_A.get_size()
    new_mat = Matrix(height, width)

    for row_idx in range(height):
      row = mat_A.get_row(row_idx)
      new_mat.set_col(row_idx, row)
    return new_mat
  @classmethod
  def add_one_row(cls, mat_A, row1_idx, row2_idx, scal_A):
    '''
    Adds scal_A * mat_A[row1] to mat_A[row2]
    '''
    row_A = Vector(mat_A.get_row(row1_idx))
    row_A = Vector.scal_mult(row_A, scal_A)

    row_B = Vector(mat_A.get_row(row2_idx))
    row_B = Vector.add(row_A, row_B).values

    mat_A.set_row(row2_idx, row_B)
    return mat_A
  @classmethod
  def scal_mult_one_row(cls, mat_A, row1, scalA):
    rowA = Vector(mat_A.get_row(row1))
    rowA = Vector.scal_mult(rowA, scalA).values
    mat_A.set_row(row1, rowA)
    return mat_A

  @classmethod
  def RREFSort(cls, mat_A, row_idx):
    '''
    Swaps rows starting at row_idx so that the row at row_idx will have all columns before row_idx be 0, and the position at row_idx will be non-zero(for row Reduction)

    Ex. [0, 0, 1, 2, ...] for row_idx 2
    Assumes that the rows above row_idx are already RREF up to row_idx in column.
    Ex [1, 0, ...]
    '''
    width, height = mat_A.get_size()
    # Select out Rows starting at row_idx
    rows_remaining = [mat_A.get_row(idx_row) for idx_row in range(row_idx, height)]
    # Find one where pos at row_idx is not 0
    rows_not_selected = []
    rows_selected = []
    for idx in range(len(rows_remaining)):
      if rows_remaining[idx][row_idx] != 0:
        if len(rows_selected) == 0:
          rows_selected += [rows_remaining[idx]]
        else:
          rows_not_selected += [rows_remaining[idx]]
      else:
        rows_not_selected += [rows_remaining[idx]]
    rows = rows_selected + rows_not_selected
    count = 0
    for idx in range(row_idx, height):
      mat_A.set_row(idx, rows[count])
      count += 1
    return mat_A
  @classmethod
  def RREF(cls, mat_A):
    '''
    Row Reduces Matrix A into Row Reduced Echelon Form
    '''
    width, height = mat_A.get_size()
    for idx_row in range(height):
        if idx_row >= width:
          break # Tall Matrix
        # Sort Matrix to Assert that row ready for row_reduce
        mat_A = cls.RREFSort(mat_A, idx_row)
        # Grab Row
        row_center = mat_A.get_row(idx_row)
        val = row_center[idx_row]
        if val == 0:
          continue # No Non-Zero Entry
        mat_A = cls.scal_mult_one_row(mat_A, idx_row, 1/val)
        for idx in range(height):
          if idx == idx_row:
            continue
          row_val = mat_A.get_row(idx)[idx_row]
          mat_A = cls.add_one_row(mat_A, idx_row, idx, -row_val)
    return mat_A
  # Augmented Variant
  @classmethod
  def AUG_RREF(cls, mat_A, aug_idx):
    '''
    Performs an Augmented RREF, reducing as usual, but not crossing the augmented line(aug_idx).
    This results in an augmented row reduction
    '''
    width, height = mat_A.get_size()
    for row_idx in range(aug_idx): # Iterate Up to the Augmented Line
      mat_A = cls.RREFSort(mat_A, row_idx)
      # Select Row
      row_selected = mat_A.get_row(row_idx)
      val = row_selected[row_idx]
      if val == 0:
        continue # No Non-zero entry
      mat_A = cls.scal_mult_one_row(mat_A, row_idx, 1 / val)
      # Iterate Through rest of rows
      for idx in range(height):
        if idx == row_idx:
          continue
        row_val = mat_A.get_row(idx)[row_idx]
        mat_A = cls.add_one_row(mat_A, row_idx, idx, -row_val)
    return mat_A
  @classmethod
  def retrieve_aug(cls, mat_A, aug_line):
    '''
    Retrieves the augmented portion of the matrix.
    '''
    width, height = mat_A.get_size()
    new_mat = Matrix(width - aug_line, height)
    cur_row = 0
    cur_col = 0
    for row in range(height):
      for col in range(aug_line, width):
        new_mat.set_value(cur_row, cur_col, mat_A.get_value(row, col))
        cur_col += 1
      cur_row += 1
      cur_col = 0
    return new_mat
  @classmethod
  def any_empty_rows(cls, mat_A):
    '''
    Returns if any empty rows exist in the matrix
    '''
    width, height = mat_A.get_size()
    for row_idx in range(height):
      row = Vector(mat_A.get_row(row_idx))
      if Vector.sum_all(row) == 0:
          return True
    return False

  @classmethod
  def is_consistent(cls, mat_A):
    '''
    Performs RREF and checks if the matrix is consistent or not.
    '''
    width, height = mat_A.get_size()
    # Row Reduce Down the Matrix
    reduced = cls.RREF(mat_A)
    return not cls.any_empty_rows(reduced)
  @classmethod
  def compute_cofactors(cls, mat_A):
      '''
      Computes all Cofactors of a Matrix, using recursion.
      '''
      assert mat_A.is_square()
      width, height = mat_A.get_size()
      New_Cofactors = Matrix(width, height)
      for row in range(height):
          for col in range(width):
              scalar =  (-1) ** (row + col + 2)
              new_mat = Matrix(width - 1, height - 1)
              count_r = 0
              count_c = 0
              for r in range(height):
                  for c in range(width):
                      if r != row and c != col:
                          w, h = new_mat.get_size()
                          new_mat.set_value(count_r, count_c, mat_A.get_value(r, c))
                          count_c += 1
                          if count_c >= height - 1:
                            count_r += 1
                            count_c = 0
              # Recursively Compute Determinant
              det = cls.recursive_determinant(new_mat) * scalar
              New_Cofactors.set_value(row, col, det)
      return New_Cofactors

  @classmethod
  def flatten(cls, vector_a):
      # Flattens the Matrix Down
      values = [item for value in vector_a.values for item in value]
      return Vector(values)
  @classmethod
  def recursive_determinant(cls, mat_A):
    '''
    Computes the Determinant of a Square Matrix, recursively.
    '''
    # Check that the matrix is square before continuing
    assert mat_A.is_square()
    width, height = mat_A.get_size()
    if width == 1:
        return mat_A.get_value(0, 0)
    else:
        # Compute CoFactors
        cofactor_mat = cls.compute_cofactors(mat_A)
        # Grab Top Row of Cofactors
        needed_cofactors = Vector(cofactor_mat.get_row(0))
        needed_coefficient = Vector(mat_A.get_row(0))
        return Vector.dot(needed_coefficient, needed_cofactors)

  @classmethod
  def compute_square_inverse(cls, mat_A):
    # Computes a Square Inverse, assuming that the matrix is square obviously.
    assert mat_A.is_square(), 'Matrix Must be Square.'
    assert cls.is_consistent(copy.deepcopy(mat_A)), "Matrix has No Inverse"
    width, height = mat_A.get_size()
    # Create Augmented Matrix
    aug_mat = Matrix(width, height)
    aug_mat.fill_identity()
    # Concatenate the matrices
    concat_mat = cls.matrix_concat_column(mat_A, aug_mat)
    # Row reduce
    row_reduced = cls.AUG_RREF(concat_mat, width)
    return cls.retrieve_aug(row_reduced, width)
  @classmethod 
  def zero_rows(cls, matrix, aug_line):
      '''
      Returns the indices of the rows where the values before aug_line are all zeros
      '''
      row_indices = []
      width, height = matrix.get_size() 
      for row_idx in range(height):
          row = matrix.get_row(row_idx)
          vec = Vector(row[:aug_line])
          if Vector.sum_all(vec) == 0:
              row_indices += [row_idx]
      return row_indices
  @classmethod
  def extract_free_vectors(cls, matrix):
    '''
    From a Row Reduced form, returns the first not free variable or None if there are no free variables
    '''
    # Check if Free Variables exist
    if not Matrix.has_free_variables(matrix):
      return None
    width, height = matrix.get_size()
    for row_idx in range(height):
      row = matrix.get_row(row_idx)
      found_idx = None
      for i in range(len(row)):
        if found_idx != None and row[i] != 0:
          return found_idx
        if row[i] == 1:
          # Index to remove if other variables exist
          found_idx = i
    raise Exception("An Error Occurred.")
  @classmethod
  def remove_empty_rows(cls, matrix):
    '''
    Removes all empty rows in a matrix
    '''
    W, H = matrix.get_size()
    valid_rows = []   
    for row_idx in range(H):
      row = Vector(matrix.get_row(row_idx))
      if Vector.sum_all(row) != 0:
        valid_rows += [row]
    return valid_rows

  @classmethod
  def has_free_variables(cls, matrix):
    width, height = matrix.get_size()
    for row_idx in range(height):
      row = Vector(matrix.get_row(row_idx))
      count = 0
      for i in row.values:
        if i != 0:
          count += 1
      if count > 1:
        return True
    return False


def main():
  matrix_A = Matrix(3, 3)
  matrix_A.list_fill([[-1, -2, 2], [2, 1, 1], [3, 4, 5]])
  print(Matrix.recursive_determinant(matrix_A))
if __name__ == '__main__':
  main()