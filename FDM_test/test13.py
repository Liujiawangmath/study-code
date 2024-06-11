import numpy as np
from scipy.special import legendre, roots_legendre
from scipy.integrate import quad
import matplotlib.pyplot as plt

# 真解
def u_exact(x):
    return np.sin(np.pi * x)

# 右端项 f(x)
def f(x):
    return (1 + np.pi**2) * np.sin(np.pi * x)

# Legendre 多项式基函数 (1-x^2) * P_n(x)
def phi(n, x):
    P_n = legendre(n)
    return (1 - x**2) * P_n(x)

# Galerkin 方法求解
def galerkin_method(N):
    A = np.zeros((N, N))
    b = np.zeros(N)
    
    for i in range(N):
        for j in range(N):
            integrand_A = lambda x: phi(i, x) * phi(j, x) - phi(i, x) * (np.pi**2) * phi(j, x)
            A[i, j] = quad(integrand_A, -1, 1)[0]
        
        integrand_b = lambda x: f(x) * phi(i, x)
        b[i] = quad(integrand_b, -1, 1)[0]

    c = np.linalg.solve(A, b)
    
    return c

# 配置法求解
def collocation_method(N):
    roots, _ = roots_legendre(N)
    A = np.zeros((N, N))
    b = np.zeros(N)
    
    for i in range(N):
        for j in range(N):
            A[i, j] = phi(j, roots[i]) - (np.pi**2) * phi(j, roots[i])
        b[i] = f(roots[i])

    c = np.linalg.solve(A, b)
    
    return c, roots

# 计算误差 L^2 范数
def compute_error(c, method='galerkin', N=10):
    error = lambda x: (u_exact(x) - sum(c[i] * phi(i, x) for i in range(N)))**2
    L2_error = np.sqrt(quad(error, -1, 1)[0])
    return L2_error

# 测试不同 N 下的误差
N_values = [4, 8, 16, 32]
errors_galerkin = []
errors_collocation = []

for N in N_values:
    c_galerkin = galerkin_method(N)
    L2_error_galerkin = compute_error(c_galerkin, method='galerkin', N=N)
    errors_galerkin.append(L2_error_galerkin)

    c_collocation, roots = collocation_method(N)
    L2_error_collocation = compute_error(c_collocation, method='collocation', N=N)
    errors_collocation.append(L2_error_collocation)

# 绘图
plt.figure(figsize=(10, 5))

plt.loglog(N_values, errors_galerkin, 'o-', label='Galerkin Method')
plt.loglog(N_values, errors_collocation, 's-', label='Collocation Method')

plt.xlabel('N')
plt.ylabel('L2 Error')
plt.title('Error Convergence of Spectral Methods')
plt.legend()
plt.grid(True, which="both", ls="--")

plt.tight_layout()
plt.show()
