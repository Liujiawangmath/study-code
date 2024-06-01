import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy.sparse.linalg import spsolve
from scipy.sparse import spdiags
from fealpy.fem import DirichletBC
from fealpy.functionspace import LagrangeFESpace as Space
from fealpy.fem import LinearElasticityOperatorIntegrator
from fealpy.fem import VectorSourceIntegrator
from fealpy.fem import VectorMassIntegrator
from fealpy.fem import BilinearForm
from fealpy.fem import LinearForm
from fealpy.decorator import cartesian
from fealpy.mesh import TriangleMesh,TetrahedronMesh
from fealpy.geometry.domain_2d import RectangleDomain

class BoxDomainData2d:
    """
    @brief 混合边界条件的线弹性问题模型
    @note 本模型假设在二维方形区域 [0,1] x [0,1] 内的线性弹性问题
    """
    def __init__(self, E=1.0, nu =0.3):
        """
        @brief 构造函数
        @param[in] E 弹性模量，默认值为 1.0
        @param[in] nu 泊松比，默认值为 0.3
        """
        self.E=E
        self.nu=nu
        
        self.lam = self.nu*self.E/((1+self.nu)*(1-2*self.nu))
        self.mu=self.E/(2*(1+self.nu))
        
    def domain(self):
        return [0,1,0,1]
        
    def init_mesh(self, n):
        """
        @brief 初始化网格
        @param[in] n 网格加密的次数，默认值为 1
        @return 返回一个初始化后的网格对象
        """
        h = 0.5
        domain = RectangleDomain()
        mesh = TriangleMesh.from_domain_distmesh(domain, h, output=False)
        mesh.uniform_refine(n)

        return mesh 

    def triangle_mesh(self):
        mesh = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=5, ny=5)

        return mesh

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = 35/13*y - 35/13*y**2 + 10/13*x - 10/13*x**2
        val[..., 1] = -25/26*(-1+2*y) * (-1+2*x)
        return val

    @cartesian
    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = x*(1-x)*y*(1-y)
        val[..., 1] = 0
        return val

    @cartesian
    def dirichlet(self, p):
        val = np.zeros((p.shape[0], 2), dtype=np.float64)
        return val

    @cartesian
    def is_dirichlet_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        flag1 = np.abs(x) < 1e-13
        flag2 = np.abs(x - 1) < 1e-13
        flagx = np.logical_or(flag1, flag2)
        flag3 = np.abs(y) < 1e-13
        flag4 = np.abs(y - 1) < 1e-13
        flagy = np.logical_or(flag3, flag4)
        flag = np.logical_or(flagx, flagy)
        return flag
    
class BoxDomainData3d():
    def __init__(self, E=1.0, nu =0.3):
        """
        @brief 构造函数
        @param[in] E 弹性模量，默认值为 1.0
        @param[in] nu 泊松比，默认值为 0.3
        """
        self.E=E
        self.nu=nu
        
        self.lam = self.nu*self.E/((1+self.nu)*(1-2*self.nu))
        self.mu=self.E/(2*(1+self.nu))
        
    def domain(self):
        return [0.0, 1, 0.0, 1, 0.0, 1]
    
    def init_mesh(self,n):
        mesh = TetrahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=5, ny=5, nz=5)
        mesh.uniform_refine(n)
        return mesh
    
    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        val = np.zeros(p.shape, dtype=np.float64)
        
        common_term = (1 - 2*x)*(1 - 2*y)*(1 - 2*z)
        factor = -(2*self.mu + self.lam)
        
        val[..., 0] = factor * common_term
        val[..., 1] = factor * common_term
        val[..., 2] = factor * common_term
        
        return val

    @cartesian
    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = x*(1-x)*y*(1-y)*z*(1-z)
        val[..., 1] = y*(1-y)*z*(1-z)*x*(1-x)
        val[..., 2] = z*(1-z)*x*(1-x)*y*(1-y)
        return val
    
    @cartesian
    def dirichlet(self, p):
        val = np.array([0.0, 0.0, 0.0])
        return val
    
    @cartesian
    def is_dirichlet_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        flag1 = np.abs(x) < 1e-12
        flag2 = np.abs(x - 1) < 1e-12
        flag3 = np.abs(y) < 1e-12
        flag4 = np.abs(y - 1) < 1e-12
        flag5 = np.abs(z) < 1e-12
        flag6 = np.abs(z - 1) < 1e-12
        return np.logical_or(np.logical_or(flag1, flag2), np.logical_or(np.logical_or(flag3, flag4), np.logical_or(flag5, flag6)))
    
    @cartesian
    def neumann(self, p, n):
        val = np.array([0.0, -50, 0.0], dtype=np.float64)
        return val
    
    @cartesian
    def is_neumann_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        flag = np.abs(y - 0.2) < 1e-13
        return flag

parser = argparse.ArgumentParser(description="单纯形网格（三角形、四面体）网格上任意次有限元方法")
parser.add_argument('--degree', default=1, type=int, help='Lagrange 有限元空间的次数, 默认为 2 次.')
parser.add_argument('--GD', default=3, type=int, help='模型问题的维数, 默认求解 2 维问题.')
parser.add_argument('--nrefine', default=2, type=int, help='初始网格加密的次数, 默认初始加密 2 次.')
parser.add_argument('--scale', default=1, type=float, help='网格变形系数，默认为 1')
parser.add_argument('--doforder', default='vdims', type=str, help='自由度排序的约定，默认为 vdims')
args = parser.parse_args()

p = args.degree
GD = args.GD
#n = args.nrefine
scale = args.scale
doforder = args.doforder
pde = BoxDomainData3d()
mu = pde.mu
lambda_ = pde.lam

mesh = pde.init_mesh(n=2)
space = Space(mesh, p=p, doforder=doforder)
uh = space.function(dim=GD)
vspace = GD * (space, )
gdof = vspace[0].number_of_global_dofs()
vgdof = gdof * GD
ldof = vspace[0].number_of_local_dofs()
vldof = ldof * GD

integrator1 = LinearElasticityOperatorIntegrator(lam=lambda_, mu=mu, q=p+3)
bform = BilinearForm(vspace)
bform.add_domain_integrator(integrator1)
KK = integrator1.assembly_cell_matrix(space=vspace)
bform.assembly()
K = bform.get_matrix()

integrator3 = VectorSourceIntegrator(f=pde.source, q=p+3)
lform = LinearForm(vspace)
lform.add_domain_integrator(integrator3)
FK = integrator3.assembly_cell_vector(space=vspace)
lform.assembly()
F = lform.get_vector()

isBdNode = mesh.ds.boundary_node_flag()
if hasattr(pde, 'dirichlet'):
    bc = DirichletBC(vspace, pde.dirichlet, threshold=isBdNode)
    K, F = bc.apply(K, F, uh)

uh.flat[:] = spsolve(K, F)
print('uh:',uh)
uh_max = np.max(uh)
print('uh_max:',uh_max)
u_exact = space.interpolate(pde.solution)
print('u_exact:',u_exact)
u_exact_max = np.max(u_exact)
print('u_exact_max:',u_exact_max)
error = mesh.error(uh,u_exact)
print('error:',error)

output = './mesh_linear/'
if not os.path.exists(output):
    os.makedirs(output)
fname = os.path.join(output, 'linear_elastic.vtu')

mesh.nodedata['u'] = uh[:, 0]
mesh.nodedata['v'] = uh[:, 1]
mesh.nodedata['W'] = uh[:, 2]
mesh.to_vtk(fname=fname)

"""
from fealpy.tools.show import showmultirate
import matplotlib.pyplot as plt

showmultirate(plt, 2, NDof, errorMatrix, errorType, propsize=20, lw=2, ms=4)
plt.xlabel('NDof')
plt.ylabel('error')
plt.tight_layout()
plt.show()
"""