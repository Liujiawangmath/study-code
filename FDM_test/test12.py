import numpy as np
import matplotlib.pyplot as plt


def solve_bvp(n):
    h = 1 / n
    x = np.linspace(0, 1, n+1)
    A = np.zeros((n+1, n+1))
    b = np.zeros(n+1)

    # 设置矩阵A和向量b
    for i in range(1, n):
        xi = x[i]
        A[i, i-1] = -1/h**2 - 10*(2*xi - 1)/(2*h)
        A[i, i] = 2/h**2 + 20
        A[i, i+1] = -1/h**2 + 10*(2*xi - 1)/(2*h)

    # 边界条件
    A[0, 0] = 1
    A[n, n] = 1
    b[0] = 1
    b[n] = 1

    # 求解线性方程组
    u = np.linalg.solve(A, b)
    
    # 真解
    u_exact = np.exp(-10 * x * (1 - x))
    
    # 计算误差
    error = np.abs(u - u_exact)
    max_error = np.max(error)
    
    return x, u, u_exact, max_error

# 测试不同网格分辨率下的误差
n_values = [10, 20, 40, 80, 160]
errors = []

for n in n_values:
    x, u, u_exact, max_error = solve_bvp(n)
    errors.append(max_error)
    print(f"n = {n}, max error = {max_error}")

# 计算收敛阶
h_values = [1.0 / n for n in n_values]
log_h = np.log(h_values)
log_errors = np.log(errors)
coeffs = np.polyfit(log_h, log_errors, 1)
convergence_rate = coeffs[0]

print(f"收敛阶:", convergence_rate)


# 绘图
plt.figure(figsize=(12, 6))

# 数值解和真解对比
plt.subplot(1, 2, 1)
plt.plot(x, u, 'o-', label='Numerical Solution')
plt.plot(x, u_exact, '-', label='Exact Solution')
plt.xlabel('x')
plt.ylabel('u')
plt.title('Numerical vs Exact Solution')
plt.legend()

# 误差与步长的关系
plt.subplot(1, 2, 2)
plt.loglog(h_values, errors, 'o-', label='Numerical Error')
plt.xlabel('Step size h')
plt.ylabel('Max Error')
plt.title('Error vs Step size')
plt.legend()
plt.grid(True, which="both", ls="--")

plt.tight_layout()
plt.show()
