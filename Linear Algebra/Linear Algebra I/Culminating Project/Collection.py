import copy
import math
from Matrix import Matrix 
from Vector import Vector
class Collection:
  '''
  Collection of Vectors, used to either compute a basis or serve as a basis.
  '''
  def __init__(self):
    self.vectors = []
    self.width = 0
    self.height = 0
  def __len__(self):
    return len(self.vectors)
  def __str__(self):
    matrix = Matrix(self.width, self.height)
    for col_idx in range(len(self.vectors)):
      matrix.set_col(col_idx, self.vectors[col_idx].values)
    print(matrix)
    return ""
  def remove_col(self, col_idx):
    self.vectors.pop(col_idx)
    self.width -= 1
  def get_size(self):
    return (self.width, self.height)
  def in_span(self, vec_A):
    '''
    Computes if a given vector is already spanned by the collection
    '''
    if isinstance(vec_A, Matrix):
      # convert to vector
      vec_A = Matrix.flatten(vec_A)
    assert isinstance(vec_A, Vector) 
    assert len(vec_A) == self.height
    # Construct a Matrix
    matrix = Matrix(self.width + 1, self.height)
    for vector_idx in range(len(self.vectors)):
      matrix.set_col(vector_idx, self.vectors[vector_idx].values)
    matrix.set_col(self.width, vec_A.values)
    # Row Reduce matrix 
    reduced = Matrix.AUG_RREF(matrix, self.width)
    # All 0 values
    row_indices = Matrix.zero_rows(reduced, self.width)
    for row_idx in row_indices:
      if reduced.get_value(row_idx, self.width) != 0:
        return False
    return True
  def matrix_as_collection(self, matrix):
    width, height = matrix.get_size()
    for col_idx in range(width):
      vec = Vector(matrix.get_col(col_idx))
      self.add_vectors(vec)
  def add_matrix(self, matrix):
    '''
    Flattens a Matrix down to a vector. Deterministic, and thus should suffice for converting a matrix collection to a vector one.
    '''
    flat_list = Matrix.flatten(matrix)
    self.add_vectors(flat_list)
  def list_add(self, vector_list):
    '''
    Converts all vectors in list into a vector and adds it to the collection
    '''
    for vector in vector_list:
      self.add_vectors(vector)
  def add_vectors(self, vector):
    if isinstance(vector, Matrix):
      self.add_matrix(vector)
      return
    assert isinstance(vector, Vector)
    self.width += 1
    if len(self.vectors) == 0:
      self.height = len(vector)
    else:
      assert self.height == len(vector), "All Vectors must be the same length"
    self.vectors += [vector]
  @classmethod
  def apply_condition(cls, mat_A, var_idx, condition):
    '''
    Applies a condition onto a vector_collection and then extracts vectors from it.

    Ex.
    x_1 = x_2 + x_3 is represented as var_idx = 0, condition = Vector([0, 1, 1])
    This calculator is no algebra solver. So please fix the condition first to be in this form: var = linear combination of other vars
    '''
    if isinstance(mat_A, (Matrix)):
      mat_A = Matrix.flatten(mat_A)
    assert isinstance(mat_A, (Vector))
    length = len(mat_A)
    # Construct a Collection of Vector
    collection = [Vector([0 for i in range(length)]) for i in range(length)]
    for i in range(length):
      collection[i].values[i] = 1
    collection.pop(0) # Remove Substituted Vector
    # Add ones into the matrix
    for i in range(1, len(condition)):
      collection[i - 1].values[var_idx] = condition.values[i]
    final_collection = Collection()
    final_collection.list_add(collection)
    return final_collection
  @classmethod
  def reduce_basis(cls, mat_A, var_idx, condition):
    '''
    Finds a Matrix for a given matrix A
    '''
    collection = Collection.apply_condition(mat_A, var_idx, condition)
    reduced_collection = Collection.reduce_collection(collection)
    return reduced_collection
  @classmethod
  def linearly_independent(cls, vector_collection):
    '''
    Computes if a given vector collection is linearly independent or not.
    returns: boolean.
    '''
    width, height = vector_collection.get_size()
    matrix = Matrix(width, height)
    for col_idx in range(len(vector_collection.vectors)):
      matrix.set_col(col_idx, vector_collection.vectors[col_idx].values)
    # Row Reduce Matrix and Check that it has no free variables
    row_reduced = Matrix.RREF(matrix)
    free_vars = Matrix.has_free_variables(row_reduced)
    return free_vars 
  @classmethod
  def reduce_collection(cls, vector_collection):
    '''
    Reduces a collection of vectors
    '''
    while Collection.reduce_RREF_collection(vector_collection):
      continue # Continue Removing Vectors from the set until fully reduced
    return vector_collection
  @classmethod
  def reduce_RREF_collection(cls, vector_collection):
    '''
    Converts a Vector Collection into a matrix and removed a dependent vector
    Returns True if vector was removed, else False.
    '''
    width, height = vector_collection.get_size()
    matrix = Matrix(width, height)
    for col_idx in range(len(vector_collection)):
      matrix.set_col(col_idx, vector_collection.vectors[col_idx].values)
    row_reduced = Matrix.RREF(matrix)
    # Extract Redundant Vector
    free_idx = Matrix.extract_free_vectors(row_reduced)
    if free_idx == None:
      # Nothing to remove
      return False
    # Remove this vector from the collection.
    vector_collection.remove_col(free_idx)
    return True

  @classmethod
  def convert_to_matrix(cls, vector_collection):
    '''
    vector_collection: Collection of vectors
    '''
    width, height = vector_collection.get_size()
    matrix = Matrix(width, height)
    for col_idx in range(width):
      matrix.set_col(col_idx, vector_collection.vectors[col_idx].values)
    return matrix

  @classmethod
  def compute_basis(cls, vector_collection):
    '''
    Takes in a vector collection, reduces it down to basis, then extends it to the whole vector space(self.height)
    '''
    copied_collection = copy.deepcopy(vector_collection)
    Collection.reduce_RREF_collection(vector_collection)
    # Append all basis vectors(RREF removes the first vectors first) 
    mat_collection = Collection.convert_to_matrix(vector_collection)
    width, height = mat_collection.get_size()

    mat_identity = Matrix(height, height)
    mat_identity.fill_identity()
    # Append the two
    concat = Matrix.matrix_concat_column(mat_identity, mat_collection)
    # Convert to Collection
    width, height = concat.get_size()
    collection = Collection()
    collection.matrix_as_collection(concat)
    # Reduce Collection
    reduced_collection = Collection.reduce_collection(collection)
    width, height = copied_collection.get_size()

    reduced_collection.vectors[-width:] = copied_collection.vectors
    return reduced_collection
def main():
  matrix = Matrix(2, 4)
  matrix.list_fill([[9, 9], [1, 10], [6, 5], [6, 2]])
  collection = Collection()
  collection.matrix_as_collection(matrix)
  print(collection)
  collected = Collection.compute_basis(collection)
  print(collected)
if __name__ == '__main__':
  main()