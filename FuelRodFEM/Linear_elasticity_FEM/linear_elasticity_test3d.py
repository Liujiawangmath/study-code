import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import spsolve
from fealpy.fem import DirichletBC
from fealpy.functionspace import LagrangeFESpace as Space
from fealpy.fem import LinearElasticityOperatorIntegrator
from fealpy.fem import VectorSourceIntegrator
from fealpy.fem import BilinearForm
from fealpy.fem import LinearForm
from fealpy.decorator import cartesian
from fealpy.mesh import  TetrahedronMesh
from fealpy.geometry.domain_2d import RectangleDomain
from scipy.sparse import spdiags
from fealpy.fem import ScalarLaplaceIntegrator
class BoxDomainData3d():
    def __init__(self, E=2.5, nu=0.25):
        self.E = E
        self.nu = nu
        self.lam = self.nu * self.E / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu = self.E / (2 * (1 + self.nu))

    def domain(self):
        return [0.0, 1, 0.0, 1, 0.0, 1]

    def init_mesh(self, n):
        mesh = TetrahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=1, ny=1, nz=1)
        mesh.uniform_refine(n)
        return mesh

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        val = np.zeros(p.shape, dtype=np.float64)

        factor1 = 400 * self.mu * (2 * y - 1) * (2 * z - 1)
        term1 = 3 * (x ** 2 - x) ** 2 * (y ** 2 - y + z ** 2 - z)
        term2 = (1 - 6 * x + 6 * x ** 2) * (y ** 2 - y) * (z ** 2 - z)
        val[..., 0] = factor1 * (term1 + term2)

        factor2 = -200 * self.mu * (2 * x - 1) * (2 * z - 1)
        term1 = 3 * (y ** 2 - y) ** 2 * (x ** 2 - x + z ** 2 - z)
        term2 = (1 - 6 * y + 6 * y ** 2) * (x ** 2 - x) * (z ** 2 - z)
        val[..., 1] = factor2 * (term1 + term2)

        factor3 = -200 * self.mu * (2 * x - 1) * (2 * y - 1)
        term1 = 3 * (z ** 2 - z) ** 2 * (x ** 2 - x + y ** 2 - y)
        term2 = (1 - 6 * z + 6 * z ** 2) * (x ** 2 - x) * (y ** 2 - y)
        val[..., 2] = factor3 * (term1 + term2)

        return val

    @cartesian
    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        val = np.zeros(p.shape, dtype=np.float64)

        val[..., 0] = 200 * self.mu * (x - x ** 2) ** 2 * (2 * y ** 3 - 3 * y ** 2 + y) * (2 * z ** 3 - 3 * z ** 2 + z)
        val[..., 1] = -100 * self.mu * (y - y ** 2) ** 2 * (2 * x ** 3 - 3 * x ** 2 + x) * (2 * z ** 3 - 3 * z ** 2 + z)
        val[..., 2] = -100 * self.mu * (z - z ** 2) ** 2 * (2 * y ** 3 - 3 * y ** 2 + y) * (2 * x ** 3 - 3 * x ** 2 + x)

        return val

    @cartesian
    def dirichlet(self, p):
        return pde.solution(p)

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
parser.add_argument('--degree', default=1, type=int, help='Lagrange 有限元空间的次数, 默认为 1 次.')
parser.add_argument('--GD', default=3, type=int, help='模型问题的维数, 默认求解 3 维问题.')
parser.add_argument('--nrefine', default=2, type=int, help='初始网格加密的次数, 默认初始加密 2 次.')
parser.add_argument('--scale', default=1, type=float, help='网格变形系数，默认为 1')
parser.add_argument('--doforder', default='vdims', type=str, help='自由度排序的约定，默认为 vdims')
args = parser.parse_args()

p = args.degree
GD = args.GD
scale = args.scale
doforder = args.doforder
pde = BoxDomainData3d()
mu = pde.mu
lambda_ = pde.lam

output = './mesh_linear/'
if not os.path.exists(output):
    os.makedirs(output)

for n in range(1):
    mesh = pde.init_mesh(n=n)
    space = Space(mesh, p=p, doforder=doforder)
    uh = space.function(dim=GD)
    vspace = GD * (space, )
    gdof = vspace[0].number_of_global_dofs()
    vgdof = gdof * GD
    ldof = vspace[0].number_of_local_dofs()
    vldof = ldof * GD

    integrator1 = LinearElasticityOperatorIntegrator(lam=lambda_, mu=mu, q=p+2)
    bform = BilinearForm(vspace)
    bform.add_domain_integrator(integrator1)
    KK = integrator1.assembly_cell_matrix(space=vspace)
    bform.assembly()
    K = bform.get_matrix()
    print(KK)

    integrator3 = VectorSourceIntegrator(f=pde.source, q=p+6)
    lform = LinearForm(vspace)
    lform.add_domain_integrator(integrator3)
    FK = integrator3.assembly_cell_vector(space=vspace)
    lform.assembly()
    F = lform.get_vector()

    isBdNode = mesh.ds.boundary_node_flag()
    if hasattr(pde, 'dirichlet'):
        dflag = vspace[0].boundary_interpolate(gD=pde.dirichlet, uh=uh,
                                               threshold=pde.is_dirichlet_boundary)
        F -= K @ uh.flat

        bdIdx = np.zeros(K.shape[0], dtype=np.int_)
        bdIdx[dflag.flat] = 1
        D0 = spdiags(1 - bdIdx, 0, K.shape[0], K.shape[0])
        D1 = spdiags(bdIdx, 0, K.shape[0], K.shape[0])
        K = D0 @ K @ D0 + D1

        F[dflag.flat] = uh.ravel()[dflag.flat]

    uh.flat[:] = spsolve(K, F)
    residual = F - K @ uh.flat
    # 计算 L2 相对范数残量
    residual_L2 = np.linalg.norm(residual, ord=2)
    F_L2 = np.linalg.norm(F, ord=2)
    residual_L2_rel = residual_L2 / F_L2
    print('相对 L2 范数残差：', residual_L2_rel)
    # 测试刚度矩阵
    u_exact = space.function(dim=GD)
    u_exact[:] = space.interpolate(pde.solution)
    u_exact_flat = u_exact.flatten()

    # 插值逼近误差计算
    F_exact = space.function(dim=GD)
    F_exact[:] = space.interpolate(pde.source)
    F_exact_flat = F_exact.flatten()

    # 计算 K @ u_exact 和 F_apm 的误差
    F_apm = K @ u_exact_flat
    error_F = np.linalg.norm(F_apm - F_exact_flat,ord=2)
    print(f'n={n}, 逼近误差：{error_F}')

    #print(f'n={n}, 误差：{error}')
    
    fname = os
