import copy 
import math
from Matrix import Matrix
from Vector import Vector
from Algebra import * # Import Algebra Related Classes.
class Basis:
  def __init__(self):
    self.basis_vectors = []
    self.width = 0
    self.height = 0
  def __len__(self):
    return self.width
  def get_size(self):
    return (self.width, self.height)
  def add_basis_vector(self, vector):
    if self.height == 0:
      self.height = len(vector)
    else:
      assert self.height == len(vector)
    self.basis_vectors += [vector]
    self.width += 1
class Coordinate:
  '''
  Class That stores coordinate information for converting under bases.
  '''
  @classmethod
  def convert_linear_mapping(cls, linear_mapping, basis_vectors):
    '''
    converts a standard matrix linear mapping to a given coordinate vector
    linear_mapping: Matrix that inputs size basis_vectors
    basis_vectors: all basis_vectors
    '''
    lin_width, lin_height = linear_mapping.get_size()
    # convert basis vectors to a matrix
    width, height = basis_vectors.get_size()

    assert linear_mapping.is_square() and lin_width == width, "Basis Vectors and Linear Mapping should have the same shape"
    matrix = Matrix(width, height) 
    for col_idx in range(width):
      matrix.set_col(col_idx, basis_vectors.basis_vectors[col_idx].values)
    basis_vec_mat = copy.deepcopy(matrix) 
    # Apply Linear Mapping to basis vectors
    applied = Matrix.matmul(linear_mapping, matrix)
    # Solve for Linear Mapping in base_B
    concat = Matrix.matrix_concat_column(basis_vec_mat, applied)
    reduced = Matrix.AUG_RREF(concat, width)
    # Extract Augmented Portion
    extracted = Matrix(lin_width, lin_height)
    count = 0
    for col_idx in range(width, lin_width + width):
      extracted.set_col(count, reduced.get_col(col_idx))
      count += 1
    return extracted


  @classmethod 
  def _sum_all(cls, vectors):
    '''
    Helper function for coordinates, performs a linear sum for all items inside of a list of vectors
    '''
    assert isinstance(vectors, (Basis))
    sum = Vector([0 for i in range(vectors.height)])
    for vector in vectors.basis_vectors:
      sum = Vector.add(sum, vector)
    return sum     

  @classmethod
  def convert_vector(cls, x_b, ordered_basis):
    '''
    Converts [x]_B and ordered Basis to x.
    '''
    length = len(x_b)
    width, height = ordered_basis.get_size() 
    assert length == width 
    sum = Vector([0 for i in range(height)])
    for i in range(length):
      sum = Vector.add(sum, Vector.scal_mult(ordered_basis.basis_vectors[i], x_b.values[i]))
    return sum
  @classmethod
  def compute_coordinate(cls, ordered_basis, x):
    '''
    Computes [x]_B given ordered basis and original vector x.
    ordered_basis: Basis Object, ordered basis 
    x: vector
    '''
    width, height = ordered_basis.get_size()
    # Construct Matrix using ordered matrix
    matrix = Matrix(width + 1, height)
    for col_idx in range(width):
      matrix.set_col(col_idx, ordered_basis.basis_vectors[col_idx].values)
    matrix.set_col(width, x.values)
    # AUG RREF to solve for coefficients
    reduced = Matrix.AUG_RREF(matrix, width)
    # Extract Vector Out
    x_b = reduced.get_col(width)
    return x_b
  @classmethod
  def change_of_basis(cls, b_coord_basis, c_coord_basis):
    '''
    Computes the change of basis matrix that converts a vector in B_coord to C_coord
    '''
    B_width, B_height = b_coord_basis.get_size()
    assert b_coord_basis.is_square() and (B_width, B_height) == c_coord_basis.get_size(), "Matrices must be square and identical in shape"
    matrix = Matrix(B_width * 2, B_height)
    for col_idx in range(B_width):
      matrix.set_col(col_idx, c_coord_basis.get_col(col_idx))
    count = B_width
    for col_idx in range(B_width):
      matrix.set_col(count, b_coord_basis.get_col(col_idx))
      count += 1
    reduced = Matrix.AUG_RREF(matrix, B_width)
    # Extract Augmented
    C_B = Matrix(B_width, B_height)
    count = 0
    for col_idx in range(B_width, B_width * 2):
      C_B.set_col(count, reduced.get_col(col_idx))
      count += 1
    return C_B

def main():
  '''
  Testing Coordinate.py
  '''
  b_coord = Matrix(3, 3)
  b_coord.list_fill([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
  c_coord = Matrix(3, 3)
  c_coord.list_fill([[1, 0, 0], [1, 1, 0], [1, 2, 1]])
  print(Coordinate.change_of_basis(b_coord, c_coord))
if __name__ == "__main__":
  main()