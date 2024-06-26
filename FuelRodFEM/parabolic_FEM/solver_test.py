import numpy as np
import os
from scipy.sparse.linalg import spsolve
from fealpy.mesh import TriangleMesh,TetrahedronMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import DiffusionIntegrator, BilinearForm, ScalarMassIntegrator, LinearForm, ScalarSourceIntegrator
from fealpy.fem.dirichlet_bc import DirichletBC
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import *



class Parabolic2dData: 
    def __init__(self,u, x, y, t, D=[0, 1, 0, 1], T=[0, 1]):
        self._domain = D 
        self._duration = T 
        self.u = lambdify([x,y,t], sympify(u))
        self.f = lambdify([x,y,t],diff(u,t,1)-diff(u,x,2)-diff(u,y,2))
        
    def domain(self):
        return self._domain

    def duration(self):
        return self._duration 

    def solution(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        return self.u(x,y,t) 

    def init_solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return self.u(x,y,self._duration[0])
        
    def source(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        return self.f(x,y,t)
    
    def gradient(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pass
       # return self.dudx(p,t)
    
    def dirichlet(self, p, t):
        return self.solution(p, t)

class Parabolic3dData:
    def __init__(self,u, x, y, z, t, D=[0, 1, 0, 1, 0, 1], T=[0, 1]):
        self._domain = D 
        self._duration = T 
        self.u = lambdify([x,y,z,t], sympify(u))
        self.f = lambdify([x,y,z,t],diff(u,t,1)-diff(u,x,2)-diff(u,y,2)-diff(u,z,2))
    
    def domain(self):
        return self._domain

    def duration(self):
        return self._duration

    def source(self,p,t):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        return self.f(x,y,z,t)

    def solution(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        return self.u(x,y,z,t) 
    
    def init_solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        return self.u(x,y,z,self._duration[0])

    def dirichlet(self, p, t):
        
        return self.solution(p, t)

    
class FuelRod3dData:
    def domain(self):
        return [0, 1, 0, 1, 0, 1]

    def duration(self):
        return [0, 1]

    def source(self,p,t):
        return 0

    def dirichlet(self, p, t):
        return np.array([500])
    
class FuelRod2dData:
    def domain(self):
        return [0, 1, 0, 1]

    def duration(self):
        return [0, 1]

    def source(self,p,t):
        return 0

    def dirichlet(self, p, t):
        return np.array([500])

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
        self.GD = self.space.geo_dimension()
        self.duration = pde.duration()
        self.nt = nt
        self.p0 = p0
        self.ficdx=ficdx
        self.cacidx=cacidx
        self.tau = (self.duration[1] - self.duration[0]) / self.nt
        self.alpha_caldding = alpha_caldding
        self.alpha_inner = alpha_inner
        self.initialize_output_directory()
        self.threshold = self.create_threshold()
        self.errors = []  # 用于存储每个时间步的误差

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


    def solve(self):
        self.p = self.space.function()
        self.p += self.p0  
        for n in range(self.nt):
            t = self.duration[0] + n * self.tau
            bform3 = LinearForm(self.space)
            # 这里应该传入后的得到的为qc
            #f=self.pde.source(node,t)
            source=lambda p: self.pde.source(p, t)
            print(source)
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
            A = self.M + self.alpha_caldding * self.K * self.tau
            b = self.M @ self.p + self.tau * self.F
            if n == 0:
                A = A
                b = b
            else:
                gD=lambda x : self.pde.dirichlet(x,t)
                ipoints = self.space.interpolation_points()
                gD=gD(ipoints[self.threshold])
                bc = DirichletBC(space=self.space, gD=gD.reshape(-1,1), threshold=self.threshold)
                A, b = bc.apply(A, b)
                self.p = spsolve(A, b)
            print(self.p)
            # 计算误差并记录
            exact_solution = self.pde.solution(self.mesh.node, t)
            error = np.linalg.norm(exact_solution - self.p.flatten('F'))
            self.errors.append(error)
            self.mesh.nodedata['temp'] = self.p.flatten('F')
            name = os.path.join(self.output, f'{self.filename}_{n:010}.vtu')
            self.mesh.to_vtk(fname=name)
        print('self.p',self.p)
        print(self.p.shape)
        
    def plot_error_over_time(self):
        plt.figure(figsize=(10, 6))
        plt.plot(np.linspace(self.duration[0], self.duration[1], self.nt), self.errors, marker='o', linestyle='-')
        plt.title('Error over Time')
        plt.xlabel('Time')
        plt.ylabel('L2 Norm of Error')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_exact_solution(self):
        t = self.duration[1]
        exact_solution = self.pde.solution(self.mesh.node, t)
        print('exact_solution',exact_solution)

        if self.GD == 2:
            plt.figure(figsize=(8, 6))
            plt.scatter(self.mesh.node[:, 0], self.mesh.node[:, 1], c=exact_solution, cmap='viridis', s=10)
            plt.colorbar()
            plt.title('Exact Solution Scatter Plot')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.tight_layout()
            plt.show()
        elif self.GD == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.mesh.node[:, 0], self.mesh.node[:, 1], self.mesh.node[:, 2], c=exact_solution, cmap='viridis', s=10)
            plt.colorbar(ax.collections[0], ax=ax)
            ax.set_title('Exact Solution Scatter Plot')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.tight_layout()
            plt.show()

    def plot_error(self):
        t = self.duration[1]
        exact_solution = self.pde.solution(self.mesh.node, t)
        numerical_solution = self.p.flatten('F')
        error = np.abs(exact_solution - numerical_solution)
        print('error',error)

        if self.GD == 2:
            plt.figure(figsize=(8, 6))
            plt.scatter(self.mesh.node[:, 0], self.mesh.node[:, 1], c=error, cmap='viridis', s=10)
            plt.colorbar()
            plt.title('Error Scatter Plot')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.tight_layout()
            plt.show()
        elif self.GD == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.mesh.node[:, 0], self.mesh.node[:, 1], self.mesh.node[:, 2], c=error, cmap='viridis', s=10)
            plt.colorbar(ax.collections[0], ax=ax)
            ax.set_title('Error Scatter Plot')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.tight_layout()
            plt.show()
        
    def plot_exact_solution_heatmap(self):
        t = self.duration[1]
        exact_solution = self.pde.solution(self.mesh.node, t)
        print('exact_solution', exact_solution)

        if self.GD == 2:
            # 假设网格是规则的
            x = self.mesh.node[:, 0]
            y = self.mesh.node[:, 1]
            z = exact_solution
            
            plt.figure(figsize=(8, 6))
            plt.tricontourf(x, y, z, levels=100, cmap='viridis')
            plt.colorbar()
            plt.title('Exact Solution Heatmap')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.tight_layout()
            plt.show()
        


    def plot_error_heatmap(self):
        t = self.duration[1]
        exact_solution = self.pde.solution(self.mesh.node, t)
        numerical_solution = self.p.flatten('F')
        error = np.abs(exact_solution - numerical_solution)
        print('error', error)

        if self.GD == 2:
            # 假设网格是规则的
            x = self.mesh.node[:, 0]
            y = self.mesh.node[:, 1]
            z = error

            plt.figure(figsize=(8, 6))
            plt.tricontourf(x, y, z, levels=100, cmap='viridis')
            plt.colorbar()
            plt.title('Error Heatmap')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.tight_layout()
            plt.show()
        


"""
# 使用示例:三维的燃料棒slover
if __name__ == "__main__":
    mm = 1e-03
    #包壳厚度
    w = 0.15 * mm
    #半圆半径
    R1 = 0.5 * mm
    #四分之一圆半径
    R2 = 1.0 * mm
    #连接处直线段
    L = 0.575 * mm
    #内部单元大小
    h = 0.5 * mm
    #棒长
    l = 20 * mm
    #螺距
    p = 40 * mm

from app.FuelRodSim.fuel_rod_mesher import FuelRodMesher 
mesher = FuelRodMesher(R1,R2,L,w,h,meshtype='segmented',modeltype='2D')
mesh = mesher.get_mesh
ficdx,cacidx = mesher.get_2D_fcidx_cacidx()
cnidx,bdnidx = mesher.get_2D_cnidx_bdnidx()
pde = FuelRod3dData()
FuelRodsolver = HeatEquationSolver(mesh, pde,640, bdnidx,300,ficdx=ficdx,cacidx=cacidx,output='./result_fuelrod2Dtest')
FuelRodsolver.solve()
"""

"""
# 使用示例:三维的燃料棒slover
if __name__ == "__main__":
    mm = 1e-03
    #包壳厚度
    w = 0.15 * mm
    #半圆半径
    R1 = 0.5 * mm
    #四分之一圆半径
    R2 = 1.0 * mm
    #连接处直线段
    L = 0.575 * mm
    #内部单元大小
    h = 0.5 * mm
    #棒长
    l = 20 * mm
    #螺距
    p = 40 * mm

from app.FuelRodSim.fuel_rod_mesher import FuelRodMesher 
mesher = FuelRodMesher(R1,R2,L,w,h,l,p,meshtype='segmented',modeltype='3D')
mesh = mesher.get_mesh
ficdx,cacidx = mesher.get_3D_fcidx_cacidx()
cnidx,bdnidx = mesher.get_3D_cnidx_bdnidx()
pde = FuelRod3dData()
FuelRodsolver = HeatEquationSolver(mesh, pde,64, bdnidx,300,ficdx=ficdx,cacidx=cacidx,output='./result_fuelrod3Dtest')
FuelRodsolver.solve()
"""

"""
#使用示例，三维箱子热传导
if __name__ == "__main__":
    nx = 20
    ny = 20
    nz = 20
from fealpy.mesh import TetrahedronMesh
mesh = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1],nx,ny,nz)
node = mesh.node
isBdNode = mesh.ds.boundary_node_flag()
pde = FuelRod3dData()
Boxsolver = HeatEquationSolver(mesh,pde,120,isBdNode,300,alpha_caldding=0.08,layered=False,output='./rusult_boxtest')
Boxsolver.solve()
"""

# 二维带真解的测试案例
pde=Parabolic2dData('exp(-2*pi**2*t)*sin(pi*x)*sin(pi*y)','x','y','t')
nx = 20
ny = 20
mesh = TriangleMesh.from_box([0, 1, 0, 1], nx,ny)
node = mesh.node
print(node.shape)
isBdNode = mesh.ds.boundary_node_flag()
p0=pde.init_solution(node) #准备一个初值
Box2dsolver = HeatEquationSolver(mesh,pde,320,isBdNode,p0=p0,alpha_caldding=1,layered=False,output='./rusult_box2dtesttest_to')
Box2dsolver.solve()
Box2dsolver.plot_exact_solution() # 绘制真解
Box2dsolver.plot_error()
Box2dsolver.plot_exact_solution_heatmap() # 绘制真解
Box2dsolver.plot_error_heatmap()
Box2dsolver.plot_error_over_time() 

"""
# 三维带真解的测试
pde=Parabolic3dData('sin(pi*x)*sin(pi*y)*sin(pi*z)*exp(-3*pi*t)','x','y','z','t')
nx = 5
ny = 5
nz = 5
mesh = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], nx, ny, nz)
node = mesh.node
isBdNode = mesh.ds.boundary_node_flag()
p0=pde.init_solution(node) #准备一个初值
Box3DSolver = HeatEquationSolver(mesh, pde, 160, isBdNode, p0=p0, alpha_caldding=1, layered=False, output='./result_box3dtest')
Box3DSolver.solve()
Box3DSolver.plot_exact_solution()
Box3DSolver.plot_error()
Box3DSolver.plot_error_over_time()
"""


