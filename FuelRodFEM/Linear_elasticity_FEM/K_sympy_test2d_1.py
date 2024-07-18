import sympy as sp

def compute_element_stiffness_matrix(phi_x_val, phi_y_val, mu, lambd, Omega):
    # 定义符号变量
    x, y = sp.symbols('x y')
    
    # 定义形函数导数矩阵的值在重心坐标处
    B_val = sp.Matrix([
        [phi_x_val.T, sp.zeros(1, len(phi_y_val)), phi_y_val.T],
        [sp.zeros(1, len(phi_x_val)), phi_y_val.T, phi_x_val.T]
    ])
    
    # 定义材料矩阵D
    D = sp.Matrix([
        [2*mu + lambd, lambd, 0],
        [lambd, 2*mu + lambd, 0],
        [0, 0, mu]
    ])
    
    # 单元刚度矩阵的积分表达式（在重心坐标处直接使用形函数导数矩阵的值）
    integrand = B_val.T * D * B_val
    
    # 对给定的区域进行积分运算
    K = sp.integrate(integrand, (x, Omega[0], Omega[1]), (y, Omega[2], Omega[3]))
    
    return K

# 示例参数：形函数在重心坐标处的导数值
phi_x_val = sp.Matrix([1, 0, -1])
phi_y_val = sp.Matrix([0, 1, -1])
mu = sp.symbols('mu')
lambd = sp.symbols('lambd')
Omega = (0, 1, 0, 1)  # 定义积分区域

# 计算单元刚度矩阵
K = compute_element_stiffness_matrix(phi_x_val, phi_y_val, mu, lambd, Omega)

# 输出单元刚度矩阵的符号表达式
sp.pprint(K)
