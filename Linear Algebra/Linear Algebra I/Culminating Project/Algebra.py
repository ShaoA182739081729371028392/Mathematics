import copy
import math
'''
Stores Algebraic Expressions, using code.

Only works in one variable, but this variable can have multiple degrees.
Can perform operations on variables, as you would with regular math
'''
class Variable:
    def __init__(self, coefficient, degree):
        self.coefficient = coefficient
        self.degree = degree
    def __add__(self, other):
        if isinstance(other, (Variable)):
            return Variable.add(self, other)
        if self.degree == 0:
            var = Variable(other, 0)
            return Variable.add(self, var)
        raise Exception("Addition Not Valid.")
    def __mul__(self, other):
        if isinstance(other, Variable):
            return Variable.multiply(self, other)
        else:
            return Variable.scal_mult(other, self)


    @classmethod
    def multiply(cls, varA, varB):
        var = Variable(varA.coefficient * varB.coefficient, varA.degree + varB.degree)
        return var
    @classmethod
    def add(cls, varA, varB):
        assert varA.degree == varA.degree
        var = Variable(varA.coefficient + varB.coefficient, varA.degree)
        return var
    @classmethod
    def scal_mult(cls, scal_mult, varA):
        var = Variable(varA.coefficient * scal_mult, varA.degree)
        return var
class AlgebraicExpression:
    def __init__(self):
        self.expression = [] 
        self.degree = -1
    def __add__(self, other):
        if isinstance(other, AlgebraicExpression):
            return AlgebraicExpression.add(self, other)
        if self.degree > -1:
            newAlg = copy.deepcopy(self)
            newAlg.expression[0] += other
            return newAlg 
    def __mul__(self, other):
        if isinstance(other, AlgebraicExpression):
            return AlgebraicExpression.alg_mult(self, other) 
        else:
            
            return AlgebraicExpression.scal_mult(self, other)
    def __str__(self):
        string = ''
        for expression in self.expression:
            if expression.degree != 0:
                string += f'{expression.coefficient}x^{expression.degree} '
            else:
                string += f'{expression.coefficient} '# Exception when degree 0
        return string
    def add_exp(self, x):
        '''
        x: Variable
        adds a variable into the expression.

        If the degree doesnt exist, it inflates the vector.

        If it does exist, it adds the variables together
        '''
        degree = x.degree
        if degree > self.degree:
            for i in range(1, degree - self.degree + 1):
                self.expression += [Variable(0, self.degree + i)]
        self.expression[degree] = Variable.add(self.expression[degree], x)
        self.degree = max(degree, self.degree)
    @classmethod
    def solve(cls, expA, x):
        '''
        Plugs in X into the equation
        '''
        sum = 0
        for VarA in expA.expression:
            sum += VarA.coefficient * x ** VarA.degree   
        return sum
    @classmethod
    def brute_force_root_search(cls, expA):
        '''
        Im not quite sure how to factor using code, so I will search for roots using bruteforce.

        Roots must be integers for this to work. Roots also must be between -100 and 100
        '''
        roots= []
        for i in range(-100, 101):
            if cls.solve(expA, i) == 0:
                roots += [i]
        return roots
    @classmethod
    def quadratic_equation(cls, expB):
        '''
        Algebraically solves for roots using the quadratic formula.
        '''
        assert expB.degree == 2
        A = expB.expression[-1].coefficient
        B = expB.expression[-2].coefficient
        C = expB.expression[-3].coefficient

        discriminant = B ** 2 - 4 * A * C

        assert discriminant >= 0, "Complex Root."
        discriminant = math.sqrt(discriminant)

        neg_b = -B
        denominator = 2 * A

        pos_root = (neg_b + discriminant) / denominator
        neg_root = (neg_b - discriminant) / denominator
        return pos_root, neg_root 
    @classmethod
    def alg_mult(cls, expA, expB):
        '''
        Performs FOIL on two given algebraic expressions 
        '''
        new_exp = AlgebraicExpression()
        for varA in expA.expression:
            for varB in expB.expression:
                new_exp.add_exp(Variable.multiply(varA, varB))
        return new_exp

    @classmethod
    def scal_mult(cls, expA, scalar):
        new_exp = copy.deepcopy(expA)
        for expression_idx in range(len(new_exp.expression)):
            expression = new_exp.expression[expression_idx]

            new_exp.expression[expression_idx] = expression * scalar
        return new_exp
    @classmethod
    def add(cls, expA, expB):
        # Simply add all values into the new expression
        exp = AlgebraicExpression()
        for A in expA.expression:
            exp.add_exp(A)
        for B in expB.expression:
            exp.add_exp(B)
        return exp
def main():
    expA = AlgebraicExpression()
    expB = AlgebraicExpression()
    
    var1 = Variable(-2, 0)
    var2 = Variable(1, 1)

    var3 = Variable(-3, 0)
    var4 = Variable(1, 1)

    expA.add_exp(var1)
    expA.add_exp(var2)

    expB.add_exp(var3)
    expB.add_exp(var4)

    print(AlgebraicExpression.alg_mult(expA, expB))
if __name__ == '__main__':
    main()