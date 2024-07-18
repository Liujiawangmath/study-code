from scipy.sparse.linalg import spsolve
from scipy.sparse import spdiags

import numpy as np
from fealpy.decorator import cartesian

from fealpy.functionspace import LagrangeFESpace as Space
from fealpy.fem import LinearElasticityOperatorIntegrator
from fealpy.fem import BilinearForm
from fealpy.fem import VectorSourceIntegrator
from fealpy.fem import LinearForm
from fealpy.mesh import TriangleMesh,TetrahedronMesh

class BoxDomainData3d():
    def __init__(self, E=1, nu =0.3):
        """
        @brief 构造函数
        @param[in] E 弹性模量，默认值为 2.5
        @param[in] nu 泊松比，默认值为 0.25
        
        """
        self.E=E
        self.nu=nu
        self.lam = self.nu*self.E/((1+self.nu)*(1-2*self.nu))# lam = 1
        self.mu=self.E/(2*(1+self.nu)) # mu = 1
        
    def domain(self):
        return [-1, 1, -1, 1, -1, 1]
    
    def init_mesh(self,nx,ny,nz):
        mesh = TetrahedronMesh.from_box(box=[-1, 1, -1, 1, -1, 1], nx=nx, ny=ny, nz=nz)
        return mesh

    """
    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        val = np.zeros(p.shape, dtype=np.float64)
        common_term = (1 - 2 * x) * (1 - 2 * y) * (1 - 2 * z)
        factor = -(2 * self.mu + self.lam)
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
        val[..., 0] = x * (1 - x) * y * (1 - y) * z * (1 - z)
        val[..., 1] = x * (1 - x) * y * (1 - y) * z * (1 - z)
        val[..., 2] = x * (1 - x) * y * (1 - y) * z * (1 - z)
        return val
    """
    

    """    
    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        val = np.zeros(p.shape, dtype=np.float64)

        factor1 = 400 * self.mu * (2*y - 1) * (2*z - 1)
        term1 = 3 * (x**2 - x)**2 * (y**2 - y + z**2 - z)
        term2 = (1 - 6*x + 6*x**2) * (y**2 - y) * (z**2 - z)
        val[..., 0] = factor1 * (term1 + term2)

        factor2 = -200 * self.mu * (2*x - 1) * (2*z - 1)
        term1 = 3 * (y**2 - y)**2 * (x**2 - x + z**2 - z)
        term2 = (1 - 6*y + 6*y**2) * (x**2 - x) * (z**2 - z)
        val[..., 1] = factor2 * (term1 + term2)

        factor3 = -200 * self.mu * (2*x - 1) * (2*y - 1)
        term1 = 3 * (z**2 - z)**2 * (x**2 - x + y**2 - y)
        term2 = (1 - 6*z + 6*z**2) * (x**2 - x) * (y**2 - y)
        val[..., 2] = factor3 * (term1 + term2)
        
        return val
    
    @cartesian
    def source(self, p):
        x1 = p[..., 0]
        x2 = p[..., 1]
        x3 = p[..., 2]
        val = np.zeros(p.shape, dtype=np.float64)

        val[..., 0] = (10 * x1**2 * x2 / 13 + 30 * x1 * x3 / 13 + 35 * x2 * x3 / 13 + 25 * x2 * x3 / 13 + 50 / 13)
        val[..., 1] = (25 * x1 * x3 / 13 + 25 * x1 * x3 / 13 + 75 * x2 * x3 / 13 + 160 / 13)
        val[..., 2] = (50 * x1 * x2 * x3 / 13 + 35 * x1 * x2 / 13 + 75 * x2 / 26 + 50 / 13)

        return val
    """
    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        val = np.zeros(p.shape, dtype=np.float64)

        # Set all components to zero
        val[..., 0] = 0
        val[..., 1] = 0
        val[..., 2] = 0

        return val

    """
    @cartesian
    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        val = np.zeros(p.shape, dtype=np.float64)

        val[..., 0] = 200 * self.mu * (x - x**2)**2 * (2*y**3 - 3*y**2 + y) * (2*z**3 - 3*z**2 + z)
        val[..., 1] = -100 * self.mu * (y - y**2)**2 * (2*x**3 - 3*x**2 + x) * (2*z**3 - 3*z**2 + z)
        val[..., 2] = -100 * self.mu * (z - z**2)**2 * (2*y**3 - 3*y**2 + y) * (2*x**3 - 3*x**2 + x)
        
        return val
    
    @cartesian
    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        val = np.zeros(p.shape, dtype=np.float64)

        # Define components of the solution
        u1 = x**2 * y * z**2 + 3 * x * y**2 * z - 2 * z
        u2 = (x + 2 * y - z)**2
        u3 = x * y * z**2 + (3 * x - y)**2

        # Assign components to the val array
        val[..., 0] = u1
        val[..., 1] = u2
        val[..., 2] = u3

        return val
    """
    @cartesian
    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        val = np.zeros(p.shape, dtype=np.float64)

        # Define components of the new solution
        u1 = 2*x**3 - 3*x*y**2 - 3*x*z**2
        u2 = 2*y**3 - 3*y*x**2 - 3*y*z**2
        u3 = 2*z**3 - 3*z*y**2 - 3*z*x**2

        # Assign components to the val array
        val[..., 0] = u1
        val[..., 1] = u2
        val[..., 2] = u3

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


pde = BoxDomainData3d()
mu = pde.mu
lambda_ = pde.lam
domain = pde.domain()
#mesh = pde.init_mesh(n=1)

p = 1
GD = 3
doforder = 'vdims'

maxit = 4
errorType = ['$|| uh - u ||_{L_2}$']
errorMatrix = np.zeros((1, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.int_)
nx = 2
ny = 2
nz = 2
mesh = pde.init_mesh(nx=nx, ny=ny, nz=nz)

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

    errorMatrix[-1, i] = mesh.error(u=uh, v=u_exact, q=p+5, power=2)
    
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