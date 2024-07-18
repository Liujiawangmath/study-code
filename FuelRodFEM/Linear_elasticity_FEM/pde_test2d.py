from sympy import symbols, lambdify, diff, Matrix, pprint, Rational
import numpy as np
class LinearElasticity2D:
    def __init__(self, u1, u2, x1, x2, E, nu, D=[0, 1, 0, 1]):
        self._domain = D
        self.u1_expr = u1
        self.u2_expr = u2
        self.u1 = lambdify([x1, x2], u1)
        self.u2 = lambdify([x1, x2], u2)
        self.E = E  # Young's modulus
        self.nu = nu  # Poisson's ratio
        
        # Define the strain tensor
        epsilon_xx = diff(u1, x1)
        epsilon_yy = diff(u2, x2)
        epsilon_xy = (diff(u1, x2) + diff(u2, x1)) / 2
        self.epsilon = Matrix([[epsilon_xx, epsilon_xy], [epsilon_xy, epsilon_yy]])
        
        # Define the stress tensor using Hooke's Law
        lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        sigma_xx = lambda_ * (epsilon_xx + epsilon_yy) + 2 * mu * epsilon_xx
        sigma_yy = lambda_ * (epsilon_xx + epsilon_yy) + 2 * mu * epsilon_yy
        sigma_xy = 2 * mu * epsilon_xy
        self.sigma = Matrix([[sigma_xx, sigma_xy], [sigma_xy, sigma_yy]])
        
        # Define the source term (body force)
        fx = -(diff(sigma_xx, x1) + diff(sigma_xy, x2))
        fy = -(diff(sigma_xy, x1) + diff(sigma_yy, x2))
        self.f = Matrix([fx, fy])
        self.f_lambdified = lambdify([x1, x2], self.f)

    def domain(self):
        return self._domain

    def solution(self, p):
        x1 = p[..., 0]
        x2 = p[..., 1]
        return np.array([self.u1(x1, x2), self.u2(x1, x2)])

    def init_solution(self, p):
        x1 = p[..., 0]
        x2 = p[..., 1]
        return np.array([self.u1(x1, x2), self.u2(x1, x2)])

    def source(self, p):
        x1 = p[..., 0]
        x2 = p[..., 1]
        fx, fy = self.f_lambdified(x1, x2)
        return np.array([fx, fy])

    def dirichlet(self, p):
        return self.solution(p)

    def print_expressions(self):
        print("Displacement field (u1):")
        pprint(self.u1_expr)
        print("\nDisplacement field (u2):")
        pprint(self.u2_expr)
        print("\nStrain tensor (epsilon):")
        pprint(self.epsilon)
        print("\nStress tensor (sigma):")
        pprint(self.sigma)
        print("\nSource term (f):")
        pprint(self.f)

# 示例使用
x1, x2 = symbols('x1 x2')
u1 = x1 * (1 - x1) * x2 * (1 - x2)
u2 = 0
E = Rational(1, 1)  # 1
nu = Rational(3, 10)  # 0.3

model = LinearElasticity2D(u1, u2, x1, x2, E, nu)

# 打印函数表达式
model.print_expressions()


