import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d import Axes3D

from fealpy.pde.linear_elasticity_model import BoxDomainData2d, BoxDomainData3d
from fealpy.mesh import TriangleMesh, TetrahedronMesh
from fealpy.functionspace import LagrangeFESpace as Space

from fealpy.fem import LinearElasticityOperatorIntegrator
from fealpy.fem import VectorSourceIntegrator
from fealpy.fem import BilinearForm
from fealpy.fem import LinearForm
from fealpy.fem import DirichletBC
from fealpy.fem import VectorNeumannBCIntegrator #TODO
from fealpy.fem import VectorMassIntegrator
from fealpy.decorator import cartesian
from fealpy.geometry.domain_2d import RectangleDomain


from timeit import default_timer as timer

class BoxDomainData2d():
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
        
    def init_mesh(self, n = 1):
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
        mesh = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=100, ny=100)

        return mesh

    @cartesian
    def source(self, p):
        """
        @brief 返回给定点的源项值 f
        @param[in] p 一个表示空间点坐标的数组
        @return 返回源项值
        """
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
        """
        @brief 返回 Dirichlet 边界上的给定点的位移
        @param[in] p 一个表示空间点坐标的数组
        @return 返回位移值，这里返回常数向量 [0.0, 0.0]
        """
        val = np.zeros((p.shape[0], 2), dtype=np.float64)
        # val = np.array([0.0, 0.0], dtype=np.float64)

        return val

    @cartesian
    def is_dirichlet_boundary(self, p):
        """
        @brief 判断给定点是否在 Dirichlet 边界上
        @param[in] p 一个表示空间点坐标的数组
        @return 如果在 Dirichlet 边界上，返回 True，否则返回 False
        """
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
    def __init__(self):
        self.L = 1
        self.W = 0.2

        self.mu = 1
        self.rho = 1

        delta = self.W/self.L
        gamma = 0.4*delta**2
        beta = 1.25

        self.lam = beta
        self.g = gamma
        self.d = np.array([0.0, 0.0, -1.0])

    def domain(self):
        return [0.0, self.L, 0.0, self.W, 0.0, self.W]

    def init_mesh(self, n=1):
        i = 2**n
        domain = self.domain()
        mesh = TetrahedronMesh(domain, nx=5*i, ny=1*i, nz=1*i)
        return mesh


    @cartesian
    def source(self, p):
#        shape = len(p.shape[:-1])*(1,) + (-1, )
        val = self.d*self.g*self.rho
        return val 
    @cartesian
    def dirichlet(self, p):
        val = np.array([0.0, 0.0, 0.0])
        return val

    @cartesian
    def is_dirichlet_boundary(self, p):
        return np.abs(p[..., 0]) < 1e-12
    
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



## 参数解析
parser = argparse.ArgumentParser(description=
        """
        单纯形网格（三角形、四面体）网格上任意次有限元方法
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--GD',
        default=2, type=int,
        help='模型问题的维数, 默认求解 2 维问题.')

parser.add_argument('--nrefine',
        default=2, type=int,
        help='初始网格加密的次数, 默认初始加密 2 次.')

parser.add_argument('--scale',
        default=1, type=float,
        help='网格变形系数，默认为 1')

parser.add_argument('--doforder',
        default='sdofs', type=str,
        help='自由度排序的约定，默认为 sdofs')

args = parser.parse_args()
p = args.degree
GD = args.GD
n = args.nrefine
scale = args.scale
doforder = args.doforder
GD = 3
if GD == 2:
    pde = BoxDomainData2d()
    mesh = pde.init_mesh(n=n)
elif GD == 3:
    pde = BoxDomainData3d()
    mesh = TetrahedronMesh.from_box()

domain = pde.domain()
NN = mesh.number_of_nodes()

# 新接口程序
# 构建双线性型，表示问题的微分形式
space = Space(mesh, p=p, doforder=doforder)
uh = space.function(dim=GD)
vspace = GD*(space, ) # 把标量空间张成向量空间
bform = BilinearForm(vspace)
bform.add_domain_integrator(LinearElasticityOperatorIntegrator(pde.lam, pde.mu))
bform.assembly()

# 构建单线性型，表示问题的源项
lform = LinearForm(vspace)
lform.add_domain_integrator(VectorSourceIntegrator(pde.source, q=1))
if hasattr(pde, 'neumann'):
    bi = VectorNeumannBCIntegrator(pde.neumann, threshold=pde.is_neumann_boundary, q=1)
    lform.add_boundary_integrator(bi)
lform.assembly()

A = bform.get_matrix()
F = lform.get_vector()

if hasattr(pde, 'dirichlet'):
    bc = DirichletBC(vspace, pde.dirichlet, threshold=pde.is_dirichlet_boundary)
    A, F = bc.apply(A, F, uh)

uh.flat[:] = spsolve(A, F)
mesh.nodedata['uh'] = uh
mesh.to_vtk(fname='linear_lfem.vtu')