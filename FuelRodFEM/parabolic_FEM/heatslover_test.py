import numpy as np
import os
from scipy.sparse.linalg import spsolve
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import DiffusionIntegrator, BilinearForm, ScalarMassIntegrator, LinearForm, ScalarSourceIntegrator
from fealpy.fem.dirichlet_bc import DirichletBC

class Parabolic2dData:
    def domain(self):
        return [0, 10, 0, 10]

    def duration(self):
        return [0, 0.1]
    
    
    def solution(self,p,t):
        pi = np.pi
        x = p[..., 0]
        y = p[..., 1]
        return np.sin(pi*x)*np.sin(pi*y)*np.exp(-2*pi*t) 
    
    def source(self, p, t):
        """
        @brief 方程右端项 

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 方程右端函数值
        """
        pi = np.pi
        x = p[..., 0]
        y = p[..., 1]
        return np.zeros(x.shape)

     
    def dirichlet(self, p,t):
        
        return self.solution(p,t)
    
class Parabolic3dData:
    def domain(self):
        return [0, 1, 0, 1, 0, 1]

    def duration(self):
        return [0, 1]

    def source(self):
        return 0

    def solution(self, p, t):
        pi = np.pi
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        return np.sin(pi * x) * np.sin(pi * y) * np.sin(pi * z) * np.exp(-3 * pi * t)

    def dirichlet(self, p, t):
        return self.solution(p, t)

class HeatEquationSolver:
    def __init__(self, mesh: TriangleMesh, pde: Parabolic2dData, nt, bdnidx, p0, alpha_caldding=4e-4, alpha_inner=8e-4, layered=True, ficdx=None ,cacidx=None,output: str = './result', filename: str = 'temp'):
        """
        Args:
            mesh (TriangleMesh): 三角形网格
            pde (ParabolicData): 双曲方程的数据
            nt (_type_): 时间迭代步长
            bdnidx (_type_): 边界条件，布尔值or索引编号
            p0 (_type_): 初始温度
            layered (bool, optional):刚度矩阵分层. Defaults to True.
            output (str, optional): 生成文件的名字. Defaults to './result'.
            filename (str, optional): Defaults to 'temp'.
        """
        self.mesh = mesh
        self.pde = pde
        self.bdnidx = bdnidx
        self.layered = layered
        self.output = output
        self.filename = filename
        self.space = LagrangeFESpace(mesh, p=1)
        self.duration = pde.duration()
        self.nt = nt
        self.p0 = p0
        self.ficdx=ficdx
        self.cacidx=cacidx
        self.tau = (self.duration[1] - self.duration[0]) / self.nt
        self.alpha_caldding = alpha_caldding
        self.alpha_inner = alpha_inner
        self.initialize_output_directory()
        self.assemble_matrices()
        self.threshold = self.create_threshold()

    def initialize_output_directory(self):
        if not os.path.exists(self.output):
            os.makedirs(self.output)

    def create_threshold(self):
        """
        可以分辨布尔数组和编号索引，如果是布尔数组直接传入，如果是索引编号转换为布尔数组
        """
        if isinstance(self.bdnidx, np.ndarray) and self.bdnidx.dtype == bool:
            return self.bdnidx
        else:
            NN = len(self.mesh.entity('node'))
            isbdnidx = np.full(NN, False, dtype=bool)
            isbdnidx[self.bdnidx] = True
            return isbdnidx

    def assemble_matrices(self,t):
        # 全局矩阵组装
        source = self.pde.source(p,t)
        bform3 = LinearForm(self.space)
        bform3.add_domain_integrator(ScalarSourceIntegrator(source, q=3))
        self.F = bform3.assembly()

        # 组装刚度矩阵
        NC = self.mesh.number_of_cells()
        alpha = np.zeros(NC)
        
        if self.layered:
            # 假设 ficdx 和 cacidx 是定义好的两个索引列表
            # 默认分层
            alpha[self.ficdx] += self.alpha_inner
            alpha[self.cacidx] += self.alpha_caldding
        else:
            # 如果不分层，使用统一的 alpha_caldding
            alpha += self.alpha_caldding

        bform = BilinearForm(self.space)
        bform.add_domain_integrator(DiffusionIntegrator(alpha, q=3))
        self.K = bform.assembly()

        # 组装质量矩阵
        bform2 = BilinearForm(self.space)
        bform2.add_domain_integrator(ScalarMassIntegrator(q=3))
        self.M = bform2.assembly()

        self.p = np.zeros_like(self.F)
        self.p += self.p0

    def apply_boundary_conditions(self, A, b, t):
        bc = DirichletBC(space=self.space, gD=self.pde.dirichlet, threshold=self.threshold)
        A, b = bc.apply(A, b)
        return A, b

    def solve(self):
        for n in range(self.nt):
            t = self.duration[0] + n * self.tau
            A = self.M + self.alpha_caldding * self.K * self.tau
            b = self.M @ self.p + self.tau * self.F
            if n == 0:
                A = A
                b = b
            else:
                A, b = self.apply_boundary_conditions(A, b, t)
            self.p = spsolve(A, b)
            self.mesh.nodedata['temp'] = self.p.flatten('F')
            name = os.path.join(self.output, f'{self.filename}_{n:010}.vtu')
            self.mesh.to_vtk(fname=name)
        print(self.p)
        print(self.p.shape)


# 二维带真解的测试案例
pde=Parabolic2dData()
nx = 5
ny = 5
mesh = TriangleMesh.from_box([0, 1, 0, 1], nx,ny)
node = mesh.node
isBdNode = mesh.ds.boundary_node_flag()
Box2dslover = HeatEquationSolver(mesh,pde,160,isBdNode,0,alpha_caldding=1,layered=False,output='./rusult_box2dtest')
Box2dslover.solve()
Box2dslover.plot_exact_solution() # 绘制真解
Box2dslover.plot_error()


"""
# 三维带真解的测试
pde=Parabolic3dData()
nx = 5
ny = 5
nz = 5
mesh = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], nx, ny, nz)
node = mesh.node
isBdNode = mesh.ds.boundary_node_flag()
Box3DSolver = HeatEquationSolver(mesh, pde, 160, isBdNode, 0, alpha_caldding=1, layered=False, output='./result_box3dtest')
Box3DSolver.solve()
Box3DSolver.plot_exact_solution()
Box3DSolver.plot_error()
"""


