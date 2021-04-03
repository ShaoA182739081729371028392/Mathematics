from Matrix import Matrix
from Vector import Vector
from Algebra import *
class EigenVector:
    '''
    Helper Functions to Compute Eigenvectors
    '''
    @classmethod
    def orig_mat(cls, diagonal_matrix, eigenvectors):
        '''
        Given the Diagonal Matrix D and Eigenvectors, compute the original matrix A
        Formulation: P^-1 A P = D
        A P = P D
        A = P D P^-1
        
        Eigenvectors: Square Matrix of eigenvectors 
        '''
        assert diagonal_matrix.is_square() and eigenvectors.is_square()
        P_inverse = Matrix.compute_square_inverse(copy.deepcopy(eigenvectors))
        return Matrix.matmul(Matrix.matmul(eigenvectors, diagonal_matrix), P_inverse)
    @classmethod
    def find_orig_mat(cls, eigenvectors, eigenvalues):
        '''
        Given eignevectors and eigenvalues, we can compute the orig matrix A.

        Formulation: P^-1 A P = D
        D = eigenvalues
        P = eigenvectors
        Solve for A
        eigenvectors: Square Matrix of vectors
        eigenvalues: list of eigen
        '''
        W, H = eigenvectors.get_size()
        assert W == H and W == len(eigenvalues)
        # Form Diagonal Matrix
        diag_matrix = Matrix(W, W) 
        for i in range(W):
            diag_matrix.set_value(i, i, eigenvalues[i])
        # Solve for A using helper
        return EigenVector.orig_mat(diag_matrix, eigenvectors)
    @classmethod
    def trace(cls, matrix):
        '''
        Computes the trill of a square
        '''
        width, height = matrix.get_size()
        trace = 0
        for i in range(width):
            trace += matrix.get_value(i, i)
        return trace
    @classmethod
    def compute_eigen(cls, matrix):
        '''
        Computes the eigenvalues and eigenvectors of matrixA, return eigen pairs of form (eigenvalue, eigenvector)
        '''
        matrix = copy.deepcopy(matrix)
        eigenvalues = EigenVector.compute_eigenvalues(matrix)
        eigenvectors = EigenVector.compute_eigenvectors(matrix, eigenvalues, filter_zeros=False)
        eigenpairs = []
        for idx in range(len(eigenvectors)):
            eigenvec = eigenvectors[idx]
            eigenval = eigenvalues[idx]
            if Vector.sum_all(eigenvec) == 0:
                eigenpairs += [(eigenval, "Invalid Eigenvector(All 0s)")]
            else:
                eigenpairs += [(eigenval, eigenvec)]
        return eigenpairs
    @classmethod
    def valid_eigenvalue(cls, eigenvalue, matrix):
        '''
        Checks if an eigenvalue is valid or not
        '''
        assert Matrix.is_square()
        W, H = matrix.get_size()
        eigenvectors = EigenVector.compute_eigenvectors(matrix, [eigenvalue], filter_zeros = True)
        return len(eigenvectors) > 0
    @classmethod
    def find_eigenvalues(cls, matrix, eigenvector):
        '''
        Finds the eigenvalue associated with the matrix, if it exists.

        matrix: Matrix
        eigenvector: Vector
        '''
        assert matrix.is_square()
        W, H = matrix.get_size()

        eigenmat = Matrix(1, W)
        eigenmat.list_fill([[val] for val in eigenvector.values])

        # Ax 
        Ax = Matrix.matmul(matrix, eigenmat)
        # Find if x is a scalar muliple of Ax
        ax_1 = Ax.get_value(0, 0)
        x_1 = eigenmat.get_value(0, 0)
        eigenvalue = ax_1 / x_1

        # Sanity Check
        if len(eigenvector) > 1:
            # Check second position to see if valid
            ax_2 = Ax.get_value(1, 0)
            x_2 = eigenmat.get_value(1, 0)
            if x_2 * eigenvalue != ax_2:
                return False
        return eigenvalue
    
    @classmethod
    def valid_eigenvector(cls, eigenvector, matrix):
        '''
        Checks if an eigenvector is valid
        eigenvector: Vector
        matrix: MatrixA
        '''
        return EigenVector.find_eigenvalues(matrix, eigenvector) != False
    @classmethod
    def compute_eigenvalues(cls, matrix):
        '''
        Computes eigenvalues of a matrix(Unique eigen values, will not return the algebraic multiplicity of each one - Need to factor for this.)
        '''
        matrix = copy.deepcopy(matrix)
        assert matrix.is_square(), "Matrix must be square."
        # Equation: det(A - LambdaI) = 0
        # Construct Algebraic Expression Matrix
        W, H = matrix.get_size()
        for i in range(W):
            x = AlgebraicExpression()
            x.add_exp(Variable(matrix.get_value(i, i), 0))
            x.add_exp(Variable(-1, 1))
            matrix.set_value(i, i, x)
        # Compute Determinant
        det = Matrix.recursive_determinant(matrix)
        # Solve for roots using brute-force(I cant factor with Python) 
        roots = AlgebraicExpression.brute_force_root_search(det)
        return roots        
    @classmethod
    def compute_eigenvectors(cls, matrix, eigenvalues, filter_zeros = True):
        '''
        Computes EigenVectors of a matrix, using the eigen values
        '''
        matrix = copy.deepcopy(matrix)
        # Formula (A - LambdaI) x = 0
        assert matrix.is_square()
        W, H = matrix.get_size()
        result = []
        for eigen in eigenvalues:
            identity = Matrix(W, H)
            identity.fill_identity()
            identity = Matrix.scal_mult(identity, eigen)
            eigen_vec = Matrix.sub(copy.deepcopy(matrix), identity)
            reduced = Matrix.RREF(eigen_vec)
            # Extract Valid Rows
            valid_rows = Matrix.remove_empty_rows(reduced)
            # Substitute Variables to get eigen vector.
            eigenvector = Vector([0 for i in range(len(valid_rows[0]))])
            for vec_idx in range(len(valid_rows)):
                # Note that x^2 = x_2 in the algebraic expression
                vec = valid_rows[vec_idx]
                right = Vector.extract_right(vec, vec_idx)
                # Check if empty: then 0
                if Vector.sum_all(right) == 0:
                    eigenvector.values[vec_idx] = 0
                else:
                    alg_exp = AlgebraicExpression()
                    for i in range(len(right)):
                        alg_exp.add_exp(Variable(-right.values[i], i + vec_idx + 1))
                    
                    eigenvector.values[vec_idx] = alg_exp
            if filter_zeros:
                # Check for fully 0 eigen vector
                if Vector.sum_all(eigenvector) == 0:
                    pass
                else:
                    result += [eigenvector]
            else:
                result += [eigenvector]
        return result

def main():
    matrix = Matrix(3, 3)
    matrix.list_fill([[3, 6, 7], [3, 3, 7], [5, 6, 5]])
    vec = Vector([1, -2, 1])
    print(EigenVector.valid_eigenvector(vec, matrix))
if __name__ == '__main__':
    main()