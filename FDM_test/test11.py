import numpy as np
import matplotlib.pyplot as plt

def wave_equation(m, n, T):
    h = 1.0 / m
    tau = T / n
    x = np.linspace(0, 1, m + 1)
    t = np.linspace(0, T, n + 1)
    u = np.zeros((m + 1, n + 1))

    # 初始条件
    for i in range(1, m):
        u[i, 0] = 0
        u[i, 1] = tau * x[i]

    # 边界条件
    for k in range(n + 1):
        u[0, k] = 0
        u[m, k] = np.sin(t[k])

    # 差分格式
    for k in range(1, n):
        for i in range(1, m):
            u[i, k + 1] = (2 * u[i, k] - u[i, k - 1] +
                           tau**2 * (u[i + 1, k] - 2 * u[i, k] + u[i - 1, k]) / h**2 +
                           tau**2 * np.sin(x[i] * t[k]))

    return x, t, u

def compute_error(m, n, T):
    x, t, u_num = wave_equation(m, n, T)
    u_exact = np.zeros_like(u_num)
    for k in range(n + 1):
        for i in range(m + 1):
            u_exact[i, k] = np.sin(x[i] * t[k])
    e = u_exact - u_num
    h = 1.0 / m
    error = np.sqrt(h * np.sum(e ** 2))
    return error

m_values = [10, 20, 40, 80, 160, 320]
errors = []

for m in m_values:
    # Adjust n to ensure tau / h > 1/2
    T = 1.0
    h = 1.0 / m
    tau = h / 2
    n = int(T / tau) -1
    
    error = compute_error(m, n, T)
    errors.append(error)
    print(f"m = {m}, n = {n}, error = {error}")

# 计算收敛阶

slopes = []
for i in range(5):
    slope = errors[i]/errors[i+1]
    slopes.append(slope)

print(f"误差比:",slopes)
plt.loglog(m_values, errors, marker='o', label='Errors')
plt.xlabel("Number of spatial points (m)")
plt.ylabel("Max error")
plt.title("Error convergence")
plt.grid(True)
plt.legend()
plt.show()

