import numpy as np
from fealpy.mesh import UniformMesh2d
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags
from sympy import *

# 定义一个 PDE 的模型类
class SinSinPDEData:
    def __init__(self,u:str,x:str = 'x',y:str = 'y',domain = [0,2,0,1]):
        self.u = lambdify((x,y),sympify(u))
        self.du_dx = lambdify((x, y), diff(u, x) )
        self.du_dy = lambdify((x, y), diff(u, y))
        self.f = lambdify((x, y), -diff(u, x, 2) - diff(u, y, 2) + sympify(u))
        self.d = domain
    
    def domain(self):
        return self.d
    
    def solution(self,p):
        x = p[...,0]
        y = p[...,1]
        return self.u(x,y)
    
    def source(self,p):
        x = p[...,0]
        y = p[...,1]
        return self.f(x,y)
    
    def gradient(self, p):
        x = p[...,0]
        y = p[...,1]
        return self.du_dx(x),self.du_dy(y)
    
    def dirichlet(self,p):
        return self.solution(p)

pde = SinSinPDEData('exp(x)*sin(2*pi*y)')
domain = pde.domain()

#网格剖分
nx = 20
ny = 20
hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny

mesh = UniformMesh2d((0, nx, 0, ny), h=(hx, hy), origin=(domain[0], domain[2]))

#画出真解图像
x = np.linspace(0, 2, 200)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
p = np.array([X, Y]).T
#print(p)
Z = pde.solution(p)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X.T, Y.T, Z, cmap='jet')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
NN = mesh.number_of_nodes()
#矩阵组装
A = mesh.laplace_operator()
B = diags([1] , [0] , shape = (NN,NN),format = 'csr')

#边界处理
uh = mesh.function()
f = mesh.interpolate(pde.source, 'node')
A, f = mesh.apply_dirichlet_bc(gD=pde.dirichlet, A=A, f=f)

#画出解得的离散图像
uh.flat[:] = spsolve(A, f)
fig = plt.figure(4)
axes = fig.add_subplot(111, projection='3d')
mesh.show_function(axes, uh.reshape(nx+1, nx+1))
plt.title("Numerical solution after processing the boundary conditions")
plt.show()

#误差计算
et = ['$|| u - u_h||_{\infty}$', '$|| u - u_h||_{0}$', '$|| u - u_h ||_{l2}$']
eu = np.zeros(len(et), dtype=np.float64)
eu[0], eu[1], eu[2] = mesh.error(pde.solution, uh)
et = np.array(et)
print(np.vstack((et, eu)))
print("----------------------------------------------------------------------")

#收敛阶
maxit = 4
em = np.zeros((3, maxit), dtype=np.float64)

for i in range(maxit):
    NN  = mesh.number_of_nodes()
    A = mesh.laplace_operator() + diags([1] , [0] , shape = (NN,NN),format = 'csr')
    uh = mesh.function()
    f = mesh.interpolate(pde.source, 'node')
    A, f = mesh.apply_dirichlet_bc(pde.dirichlet, A, f)
    uh.flat[:] = spsolve(A, f)
    uh.reshape(nx+1,ny+1)
    fig = plt.figure(4)
    axes = fig.add_subplot(111, projection='3d')
    node = mesh.entity('node')
    uI = pde.solution(node).reshape(nx+1 ,ny+1)
    e = uh - uI
    mesh.show_function(axes, e)
    plt.show()
    em[0, i], em[1, i], em[2, i] = mesh.error(pde.solution, uh)

    if i < maxit:
        mesh.uniform_refine()
        nx = 2*nx
        ny = 2*ny
print("em:\n", em)
print("----------------------------------------------------------------------")
print("em_ratio:", em[:, 0:-1]/em[:, 1:])