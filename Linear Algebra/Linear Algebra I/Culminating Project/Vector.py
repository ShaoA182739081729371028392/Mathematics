from Algebra import *
import copy
import math
class Vector:
  '''
  Holds necessary Functions for Vectors
  '''
  def __init__(self, values):
    self.values = values
  def __add__(self, other):
    if isinstance(other, Vector):
      return Vector.add(self, other)
    else:
      vector = Vector([other for i in range(len(self))])
      return Vector.add(self, vector)
  def __mul__(self, other):
    return Vector.scal_mult(self, other)
  def __str__(self):
    return str(self.values)
  def __len__(self):
    return len(self.values)
  def get_euclidean_len(self, squared = True):
    '''
    Computes the actual length of a vector
    '''
    val = self.dot(Vector(self.values), Vector(self.values))
    if not squared:
      return math.sqrt(val)
    return val
  @classmethod
  def add(cls, vector_a, vector_b):
    length = len(vector_a)
    assert length == len(vector_b)
    values = []
    for idx in range(length):
      values += [vector_a.values[idx] + vector_b.values[idx]]
    return Vector(values)
  @classmethod
  def sub(cls, vector_a, vector_b):
    length = len(vector_a)
    assert length == len(vector_b)
    values = []
    for idx in range(length):
      values += [vector_a.values[idx] - vector_b.values[idx]]
    return Vector(values)
  @classmethod
  def sum_all(cls, vector_a):
    summed = 0
    for i in vector_a.values:
      if isinstance(summed, AlgebraicExpression):
        summed = summed + i
      else:
        summed = i + summed
    return summed
  @classmethod
  def comparison(cls, vector_a, vector_b):
    length = len(vector_a)
    if len(vector_b) != length:
      return False
    for i in range(length):
      if vector_a.values[i] != vector_b.values[i]:
        return False
    return True
  @classmethod
  def scal_mult(cls, vector_a, scalar):
    values =[]
    for idx in range(len(vector_a)):
      values += [vector_a.values[idx] * scalar]
    return Vector(values)
  @classmethod
  def dot(cls, vector_a, vector_b):
    length = len(vector_a)
    assert length == len(vector_b)
    sum = 0
    for idx in range(length):
      if isinstance(vector_b.values[idx], AlgebraicExpression):
        mult = vector_b.values[idx] * vector_a.values[idx]
      else:
        mult = vector_a.values[idx] * vector_b.values[idx]
      if isinstance(sum, AlgebraicExpression): 
        sum = sum + mult
      else:
        sum = mult + sum 
    return sum
  @classmethod
  def extract_right(cls, vec_a, idx):
    '''
    Extracts all values to the right of idx in the vector and returns a new vector
    '''
    vals = []
    for val_idx in range(idx + 1, len(vec_a)):
      vals += [vec_a.values[val_idx]]
    return Vector(vals)
  @classmethod
  def proj(cls, vector_a, vector_b):
    '''
    Projects vector_a onto vector_b
    Formula: ((b dot a)/mag(a)^2 )a
    '''
    dotted = cls.dot(vector_a, vector_b)
    scaled_down = dotted * vector_a.get_euclidean_len(squared = True)
    return cls.scal_mult(vector_a, scaled_down)
    
