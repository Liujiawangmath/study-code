from sympy import symbols, lambdify, diff, Matrix, pprint, Rational
import numpy as np

class LinearElasticity3D:
    def __init__(self, u1, u2, u3, x1, x2, x3, E, nu, mu, D=[0, 1, 0, 1, 0, 1]):
        self._domain = D
        self.u1_expr = u1
        self.u2_expr = u2
        self.u3_expr = u3
        self.u1 = lambdify([x1, x2, x3], u1)
        self.u2 = lambdify([x1, x2, x3], u2)
        self.u3 = lambdify([x1, x2, x3], u3)
        self.E = E  # 杨氏模量
        self.nu = nu  # 泊松比
        self.mu = mu  # 剪切模量

        # 定义应变张量
        epsilon_xx = diff(u1, x1)
        epsilon_yy = diff(u2, x2)
        epsilon_zz = diff(u3, x3)
        epsilon_xy = (diff(u1, x2) + diff(u2, x1)) / 2
        epsilon_xz = (diff(u1, x3) + diff(u3, x1)) / 2
        epsilon_yz = (diff(u2, x3) + diff(u3, x2)) / 2
        self.epsilon = Matrix([
            [epsilon_xx, epsilon_xy, epsilon_xz],
            [epsilon_xy, epsilon_yy, epsilon_yz],
            [epsilon_xz, epsilon_yz, epsilon_zz]
        ])

        # 使用胡克定律定义应力张量
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

        # 定义源项（体力）
        fx = diff(sigma_xx, x1) + diff(sigma_xy, x2) + diff(sigma_xz, x3)
        fy = diff(sigma_xy, x1) + diff(sigma_yy, x2) + diff(sigma_yz, x3)
        fz = diff(sigma_xz, x1) + diff(sigma_yz, x2) + diff(sigma_zz, x3)
        self.f = Matrix([fx, fy, fz])
        self.f_lambdified = lambdify([x1, x2, x3], self.f)

    def domain(self):
        """
        返回定义域
        """
        return self._domain

    def solution(self, p):
        """
        计算解
        """
        x1 = p[..., 0]
        x2 = p[..., 1]
        x3 = p[..., 2]
        return np.array([self.u1(x1, x2, x3), self.u2(x1, x2, x3), self.u3(x1, x2, x3)])

    def init_solution(self, p):
        """
        计算初始解
        """
        x1 = p[..., 0]
        x2 = p[..., 1]
        x3 = p[..., 2]
        return np.array([self.u1(x1, x2, x3), self.u2(x1, x2, x3), self.u3(x1, x2, x3)])

    def source(self, p):
        """
        计算源项
        """
        x1 = p[..., 0]
        x2 = p[..., 1]
        x3 = p[..., 2]
        fx, fy, fz = self.f_lambdified(x1, x2, x3)
        return np.array([fx, fy, fz])

    def dirichlet(self, p):
        """
        计算迪利克雷边界条件
        """
        return self.solution(p)

    def print_expressions(self):
        """
        打印函数表达式
        """
        print("位移场 (u1):")
        pprint(self.u1_expr)
        print("\n位移场 (u2):")
        pprint(self.u2_expr)
        print("\n位移场 (u3):")
        pprint(self.u3_expr)
        print("\n源项 (f):")
        pprint(self.f)
"""
# 示例使用
x1, x2, x3, mu = symbols('x1 x2 x3 mu')
u1 = x1**2 * x2 * x3**2 + 3 * x1 * x2**2 * x3 - 2 * x3
u2 = (x1 + 2 * x2 - x3)**2
u3 = (3 * x1 - x2)**2 + x1 * x2 * x3**2
E = Rational(1, 1)   
nu = Rational(3, 10)  


model = LinearElasticity3D(u1, u2, u3, x1, x2, x3, E, nu, mu)

# 打印函数表达式
model.print_expressions()
"""
# 示例使用
x1, x2, x3, mu = symbols('x1 x2 x3 mu')
u1 = 2*x1**3 - 3*x1*x2**2 - 3*x1*x3**2
u2 = 2*x2**3 - 3*x2*x1**2 - 3*x2*x3**2
u3 = 2*x3**3 - 3*x3*x2**2 - 3*x3*x1**2
E = Rational(1, 1)   
nu = Rational(1, 4)  

model = LinearElasticity3D(u1, u2, u3, x1, x2, x3, E, nu, mu)

# 打印函数表达式
model.print_expressions()







