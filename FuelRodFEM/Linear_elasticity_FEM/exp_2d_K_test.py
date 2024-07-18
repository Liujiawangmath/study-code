from scipy.sparse.linalg import spsolve
from scipy.sparse import spdiags

import numpy as np
from fealpy.decorator import cartesian
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace as Space
from fealpy.fem import LinearElasticityOperatorIntegrator
from fealpy.fem import BilinearForm
from fealpy.fem import VectorSourceIntegrator
from fealpy.fem import LinearForm
from fealpy.geometry.domain_2d import RectangleDomain

class BoxDomainData2d:
    """
    @brief 混合边界条件的线弹性问题模型
    @note 本模型假设在二维方形区域 [0,1] x [0,1] 内的线性弹性问题
    """
    def __init__(self, E=1, nu =0.3):
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

    def triangle_mesh(self,nx,ny):
        mesh = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=nx, ny=ny)
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
        
        return pde.solution(p)

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

pde = BoxDomainData2d()
mu = pde.mu
lambda_ = pde.lam
#mesh = pde.init_mesh(h=0.5)
nx = 2
ny = 2
mesh = pde.triangle_mesh(nx=nx, ny=ny)

import matplotlib.pyplot as plt
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.show()

p = 1
GD = 2
doforder = 'vdims'

maxit = 6
errorType = ['$|| Ku - F ||_{L_2}$',
             '$|| uh - u ||_{L_2}$']
errorMatrix = np.zeros((2, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.int_)
nx = 2
ny = 2
mesh = pde.triangle_mesh(nx=nx, ny=ny)

for i in range(maxit):
    space = Space(mesh, p=p, doforder=doforder)
    
    u_exact = space.interpolate(pde.solution)
    uh = space.function(dim=GD)
    vspace = GD*(space, )
    gdof = vspace[0].number_of_global_dofs()
    vgdof = gdof * GD
    print("vgdof:", vgdof)
    ldof = vspace[0].number_of_local_dofs()
    vldof = ldof * GD
    NDof[i] = vspace[0].number_of_global_dofs()

    integrator1 = LinearElasticityOperatorIntegrator(
                lam=lambda_, mu=mu, q=p+5
                                                    )
    bform = BilinearForm(vspace)
    bform.add_domain_integrator(integrator1)
    KK = integrator1.assembly_cell_matrix(space=vspace)
    K = bform.assembly()

    F_exact = space.interpolate(pde.source)
    F_approx_computed = space.function(dim=GD)
    F_approx_computed.flat[:] = K @ u_exact.flat
    
    integrator3 = VectorSourceIntegrator(f=pde.source, q=p+5)
    lform = LinearForm(vspace)
    lform.add_domain_integrator(integrator3)
    FK = integrator3.assembly_cell_vector(space = vspace)
    F = lform.assembly()
    F_exact_computed = F

    cell_area = mesh.entity_measure('cell')
    h = nx * 2 ** i
    errorMatrix[0, i] = np.linalg.norm(F_approx_computed.reshape(-1) \
                                       - F_exact_computed, ord=1) 
    #errorMatrix[0, i] = mesh.error(u=F_approx_computed, v=F_exact_computed, q=p+5, power=2)

    dflag = vspace[0].boundary_interpolate(
        gD=pde.dirichlet, uh=uh,
        threshold=pde.is_dirichlet_boundary)
    ipoints = vspace[0].interpolation_points()
    uh[dflag] = pde.dirichlet(ipoints[dflag])
    
    F -= K@uh.flat
    F[dflag.flat] = uh.ravel()[dflag.flat]

    bdIdx = np.zeros(K.shape[0], dtype=np.int_)
    bdIdx[dflag.flat] = 1
    D0 = spdiags(1-bdIdx, 0, K.shape[0], K.shape[0])
    D1 = spdiags(bdIdx, 0, K.shape[0], K.shape[0])
    K = D0@K@D0 + D1

    uh.flat[:] = spsolve(K, F)

    errorMatrix[1, i] = mesh.error(u=uh, v=u_exact, q=p+5, power=2)
    
    if i < maxit-1:
        mesh.uniform_refine()

print("errorMatrix:\n", errorMatrix)
print("ratio:\n", errorMatrix[:, 0:-1]/errorMatrix[:, 1:])

from fealpy.tools.show import showmultirate
import matplotlib.pyplot as plt

showmultirate(plt, 2, NDof, errorMatrix, errorType, propsize=20, lw=2, ms=4)
plt.xlabel('NDof')
plt.ylabel('error')
plt.tight_layout()
plt.show()