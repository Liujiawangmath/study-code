import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.sparse.linalg import spsolve
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import DiffusionIntegrator
from fealpy.fem import BilinearForm
from fealpy.fem import ScalarMassIntegrator
from fealpy.fem import LinearForm
from fealpy.fem import ScalarSourceIntegrator
class ParabolicData:
    
    
    def domain(self):
        """
        @brief 空间区间
        """
        return [0, 10, 0, 10, 0, 10]

    def duration(self):
        """
        @brief 时间区间
        """
        return [0, 100]
    
    def source(self):
        
        return 0
    
    def dirichlet(self, p):
        """
        @brief 返回 Dirichlet 边界上的给定点的位移
        @param[in] p 一个表示空间点坐标的数组
        @return 返回位移值，这里返回常数向量 [0.0, 0.0]
        """
        return 75
    
    
pde=ParabolicData()
domain=pde.domain()
source=pde.source()
nx = 10
ny = 10
nz = 10
hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny
hz = (domain[5]-domain[4])/nz

mesh = TriangleMesh()
node = mesh.node
isBdNode = mesh.ds.boundary_node_flag()
print(isBdNode)

import matplotlib.pyplot as plt
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_cell(axes, showindex=True, color='k', marker='s', markersize=2, fontsize=8, fontcolor='k')
mesh.find_node(axes, showindex=True, color='r', marker='o', markersize=2, fontsize=8, fontcolor='r')
#plt.show()

# 时间离散
duration = pde.duration()
nt = 640
tau = (duration[1] - duration[0])/nt 
print("网比r_x:", tau/(hx**2)) # 0.25
print("网比r_y:", tau/(hy**2)) # 0.25

# 基函数
space = LagrangeFESpace(mesh, p=1)
qf = mesh.integrator(3) 
bcs, ws = qf.get_quadrature_points_and_weights()
phi=space.basis(bcs)


# 维数
GD=space.geo_dimension()


#组装刚度矩阵
bform = BilinearForm(space)
bform.add_domain_integrator(DiffusionIntegrator(q=3))
K = bform.assembly()
print(K)

#组装质量矩阵
bform2=BilinearForm(space)
bform2.add_domain_integrator(ScalarMassIntegrator(q=3))
M=bform2.assembly()
print(M)

bform3=LinearForm(space)
bform3.add_domain_integrator(ScalarSourceIntegrator(source,q=3))
F=bform3.assembly()
print(F)
print(F.shape)

p=np.zeros_like(F)
p+=37
alpha=1.98e-2

import os
output = './result3d'
filename = 'temp3d'
# Check if the directory exists, if not, create it
if not os.path.exists(output):
    os.makedirs(output)

for n in range(nt):
    t = duration[0] + n*tau
    A = M + alpha*K*tau
    b = M @ p + tau*F
    p=spsolve(A,b)
    # Dirichlet边界条件
    p[isBdNode] = pde.dirichlet(node)
    mesh.nodedata['temp3d'] = p.flatten('F')
    name = os.path.join(output, f'{filename}_{n:010}.vtu')
    mesh.to_vtk(fname=name)
    
print(p)
print(p.shape)
