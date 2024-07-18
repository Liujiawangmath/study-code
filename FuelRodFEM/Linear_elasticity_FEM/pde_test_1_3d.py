from sympy import symbols, lambdify, diff, Matrix, pprint, Rational
from sympy.vector import CoordSys3D
import numpy as np

class LinearElasticity3D:
    def __init__(self, u1, u2, u3, x, y, z, E, nu, mu):
        self.u1_expr = u1
        self.u2_expr = u2
        self.u3_expr = u3
        self.u1 = lambdify([x, y, z], u1)
        self.u2 = lambdify([x, y, z], u2)
        self.u3 = lambdify([x, y, z], u3)
        self.E = E  # Young's modulus
        self.nu = nu  # Poisson's ratio
        self.mu = mu  # Shear modulus

        # Define the strain tensor
        epsilon_xx = diff(u1, x)
        epsilon_yy = diff(u2, y)
        epsilon_zz = diff(u3, z)
        epsilon_xy = (diff(u1, y) + diff(u2, x)) / 2
        epsilon_xz = (diff(u1, z) + diff(u3, x)) / 2
        epsilon_yz = (diff(u2, z) + diff(u3, y)) / 2
        self.epsilon = Matrix([
            [epsilon_xx, epsilon_xy, epsilon_xz],
            [epsilon_xy, epsilon_yy, epsilon_yz],
            [epsilon_xz, epsilon_yz, epsilon_zz]
        ])

        # Define the stress tensor using Hooke's Law
        lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        sigma_xx = lambda_ * (epsilon_xx + epsilon_yy + epsilon_zz) + 2 * mu * epsilon_xx
        sigma_yy = lambda_ * (epsilon_xx + epsilon_yy + epsilon_zz) + 2 * mu * epsilon_yy
        sigma_zz = lambda_ * (epsilon_xx + epsilon_yy + epsilon_zz) + 2 * mu * epsilon_zz
        sigma_xy = 2 * mu * epsilon_xy
        sigma_xz = 2 * mu * epsilon_xz
        sigma_yz = 2 * mu * epsilon_yz
        self.sigma = Matrix([
            [sigma_xx, sigma_xy, sigma_xz],
            [sigma_xy, sigma_yy, sigma_yz],
            [sigma_xz, sigma_yz, sigma_zz]
        ])

        # Define the source term (body force)
        fx = diff(sigma_xx, x) + diff(sigma_xy, y) + diff(sigma_xz, z)
        fy = diff(sigma_xy, x) + diff(sigma_yy, y) + diff(sigma_yz, z)
        fz = diff(sigma_xz, x) + diff(sigma_yz, y) + diff(sigma_zz, z)
        self.f = Matrix([fx, fy, fz])
        self.f_lambdified = lambdify([x, y, z], self.f)

    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        return np.array([self.u1(x, y, z), self.u2(x, y, z), self.u3(x, y, z)])

    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        fx, fy, fz = self.f_lambdified(x, y, z)
        return np.array([fx, fy, fz])

    def print_expressions(self):
        print("Displacement field (u1):")
        pprint(self.u1_expr)
        print("\nDisplacement field (u2):")
        pprint(self.u2_expr)
        print("\nDisplacement field (u3):")
        pprint(self.u3_expr)
        print("\nSource term (f):")
        pprint(self.f)

# 示例使用
x, y, z, mu = symbols('x y z mu')
u1 = x * (1 - x) * y * (1 - y) * z * (1 - z)
u2 = y * (1 - y) * z * (1 - z) * x * (1 - x)
u3 = z * (1 - z) * x * (1 - x) * y * (1 - y)
E = Rational(1, 1)   # 210 GPa
nu = Rational(3, 10)  # 0.3

model = LinearElasticity3D(u1, u2, u3, x, y, z, E, nu, mu)

# 打印函数表达式
model.print_expressions()
