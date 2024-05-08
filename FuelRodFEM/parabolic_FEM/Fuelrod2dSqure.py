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

def from_fuel_rod_gmsh(R1,R2,L,w,h,meshtype='normal',square='all'):
    """
    Generate a trangle mesh for a fuel-rod region by gmsh

    @param R1 The radius of semicircles
    @param R2 The radius of quarter circles
    @param L The length of straight segments
    @param w The thickness of caldding
    @param h Parameter controlling mesh density
    @param meshtype Choose whether to add mesh refinement at the boundary
    @return TriangleMesh instance
    """
    import gmsh
    import numpy as np
    from fealpy.mesh import TriangleMesh
    gmsh.initialize()
    gmsh.model.add("fuel_rod_2D")

    # 内部单元大小
    Lc1 = h
    # 包壳单元大小
    Lc2 = h/2.5


    factory = gmsh.model.geo
    # 外圈点
    factory.addPoint( -R1 -R2 -L, 0 , 0 , Lc2 , 1 )#圆心1
    factory.addPoint( -R1 -R2 -L, -R1 , 0 , Lc2 , 2)
    factory.addPoint( -R1 -R2 , -R1 , 0 , Lc2 , 3)
    factory.addPoint( -R1 -R2 , -R1 -R2 , 0 , Lc2 , 4)#圆心2
    factory.addPoint( -R1 , -R1 -R2 , 0 , Lc2 , 5)
    factory.addPoint( -R1 , -R1 -R2 -L , 0 , Lc2 , 6)
    factory.addPoint( 0 , -R1 -R2 -L , 0 , Lc2 , 7)#圆心3
    factory.addPoint( R1 , -R1 -R2 -L , 0 , Lc2 , 8)
    factory.addPoint( R1 , -R1 -R2 , 0 , Lc2 , 9)
    factory.addPoint( R1 +R2 , -R1 -R2 , 0, Lc2 , 10)#圆心4
    factory.addPoint( R1 +R2 , -R1 , 0 , Lc2 , 11) 
    factory.addPoint( R1 +R2 +L , -R1 , 0 , Lc2 , 12)
    factory.addPoint( R1 +R2 +L , 0 , 0 , Lc2 , 13)#圆心5
    factory.addPoint( R1 +R2 +L , R1 , 0 , Lc2 , 14)
    factory.addPoint( R1 +R2 , R1 , 0 , Lc2 , 15)
    factory.addPoint( R1 +R2 , R1 +R2 , 0 , Lc2 , 16)#圆心6
    factory.addPoint( R1 , R1 +R2 , 0 , Lc2 , 17)
    factory.addPoint( R1 , R1 +R2 +L , 0 , Lc2 , 18)
    factory.addPoint( 0 , R1 +R2 +L , 0 , Lc2 , 19)#圆心7
    factory.addPoint( -R1 , R1 +R2 +L , 0 , Lc2 , 20)
    factory.addPoint( -R1 , R1 +R2 , 0 , Lc2 , 21)
    factory.addPoint( -R1 -R2 , R1 +R2 , 0 , Lc2 , 22)#圆心8
    factory.addPoint( -R1 -R2 , R1 , 0 , Lc2 , 23)
    factory.addPoint( -R1 -R2 -L , R1 , 0 , Lc2 , 24)

    # 外圈线
    line_list_out = []
    for i in range(8):
        if i == 0:
            factory.addCircleArc(24 , 3*i+1 , 3*i+2, 2*i+1)
            factory.addLine( 3*i+2 , 3*i+3 , 2*(i+1) )
        else:
            factory.addCircleArc(3*i , 3*i+1 , 3*i+2 , 2*i+1)
            factory.addLine( 3*i+2 , 3*i+3 , 2*(i+1) )
        # 填充线环中的线
        line_list_out.append(2*i+1)
        line_list_out.append(2*(i+1))
    # 生成外圈线环
    factory.addCurveLoop(line_list_out,17)

    # 内圈点
    factory.addPoint( -R1 -R2 -L, -R1 +w , 0 , Lc1 , 25)
    factory.addPoint( -R1 -R2 , -R1 +w , 0 , Lc1 , 26)
    factory.addPoint( -R1 +w , -R1 -R2 , 0 , Lc1 , 27)
    factory.addPoint( -R1 +w , -R1 -R2 -L , 0 , Lc1 , 28)
    factory.addPoint( R1 -w , -R1 -R2 -L , 0 , Lc1 , 29)
    factory.addPoint( R1 -w , -R1 -R2 , 0 , Lc1 , 30)
    factory.addPoint( R1 +R2 , -R1 +w , 0 , Lc1 , 31) 
    factory.addPoint( R1 +R2 +L , -R1 +w , 0 , Lc1 , 32)
    factory.addPoint( R1 +R2 +L , R1 -w , 0 , Lc1 , 33)
    factory.addPoint( R1 +R2 , R1 -w , 0 , Lc1 , 34)
    factory.addPoint( R1 -w , R1 +R2 , 0 , Lc1 , 35)
    factory.addPoint( R1 -w , R1 +R2 +L , 0 , Lc1 , 36)
    factory.addPoint( -R1 +w , R1 +R2 +L , 0 , Lc1 , 37)
    factory.addPoint( -R1 +w , R1 +R2 , 0 , Lc1 , 38)
    factory.addPoint( -R1 -R2 , R1 -w, 0 , Lc1 , 39)
    factory.addPoint( -R1 -R2 -L , R1 -w, 0 , Lc1 , 40)

    # 内圈线
    line_list_in = []
    for j in range(8):
        if j == 0:
            factory.addCircleArc(40 , 3*j+1 , 25+2*j , 18+2*j)
            factory.addLine(25+2*j , 26+2*j , 19+2*j)
        else:
            factory.addCircleArc(24+2*j , 3*j+1 , 25+2*j, 18+2*j)
            factory.addLine(25+2*j , 26+2*j , 19+2*j)
        line_list_in.append(18+2*j)
        line_list_in.append(19+2*j)
    
    def refinment():
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmmf = gmsh.model.mesh.field
        gmmf.add("Distance",1)
        gmmf.setNumbers(1, "CurvesList",line_list_in)
        gmmf.setNumber(1,"Sampling",1000)
        gmmf.add("Threshold",2)
        gmmf.setNumber(2, "InField", 1)
        gmmf.setNumber(2, "SizeMin", Lc1/5)
        gmmf.setNumber(2, "SizeMax", Lc1)
        gmmf.setNumber(2, "DistMin", w)
        gmmf.setNumber(2, "DistMax", 2*w)
        gmmf.setAsBackgroundMesh(2)

    # 生成内圈线环  
    factory.addCurveLoop(line_list_in,34)
    if square == 'all':
        # 内圈面
        factory.addPlaneSurface([34],35)
        # 包壳截面
        factory.addPlaneSurface([17, 34],36)
        factory.synchronize()

    elif square == 'inner':
        # 内圈面
        factory.addPlaneSurface([34],35)
        factory.synchronize()

    elif square == 'caldding':
        # 包壳截面
        factory.addPlaneSurface([17, 34],36)
        factory.synchronize()

    if meshtype == 'refine':
            refinment()
    #生成网格
    gmsh.model.mesh.generate(2)

    # 获取节点信息
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()

    node = np.array(node_coords, dtype=np.float64).reshape(-1, 3)[:, 0:2].copy()

    # 获取三角形单元信息
    cell_type = 2  # 三角形单元的类型编号为 2
    cell_tags, cell_connectivity = gmsh.model.mesh.getElementsByType(cell_type)
    cell = np.array(cell_connectivity, dtype=np.int_).reshape(-1, 3) -1

    gmsh.finalize()
    print(f"Number of nodes: {node.shape[0]}")
    print(f"Number of cells: {cell.shape[0]}")

    NN = len(node)
    isValidNode = np.zeros(NN, dtype=np.bool_)
    isValidNode[cell] = True
    node = node[isValidNode]
    idxMap = np.zeros(NN, dtype=cell.dtype)
    idxMap[isValidNode] = range(isValidNode.sum())
    cell = idxMap[cell]

    return TriangleMesh(node,cell)

class ParabolicData:
    
    
    def domain(self):
        """
        @brief 空间区间
        """
        return [0, 10, 0, 10]

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
        return 500

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
h = 0.0003

mesh = from_fuel_rod_gmsh(R1,R2,L,w,h,meshtype='normal',square='caldding')

pde=ParabolicData()
source=pde.source()


node = mesh.node
isBdNode = mesh.ds.boundary_node_flag()


# 时间离散
duration = pde.duration()
nt = 640
tau = (duration[1] - duration[0])/nt 


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


#组装质量矩阵
bform2=BilinearForm(space)
bform2.add_domain_integrator(ScalarMassIntegrator(q=3))
M=bform2.assembly()


bform3=LinearForm(space)
bform3.add_domain_integrator(ScalarSourceIntegrator(source,q=3))
F=bform3.assembly()


p=np.zeros_like(F)
p+=300
alpha=4

    
import matplotlib.patches as mpatches

# 获取边界节点和非边界节点的索引
boundary_nodes = np.where(isBdNode)[0]
interior_nodes = np.where(~isBdNode)[0]

# 为边界节点和非边界节点分配不同颜色
colors = ['red' if i in boundary_nodes else 'blue' for i in range(len(node))]

# 绘制网格并着色节点
fig, ax = plt.subplots(figsize=(8, 6))
mesh.add_plot(ax)
mesh.find_cell(ax, showindex=False)

# 在图中添加边界节点和非边界节点，使用不同颜色
for idx, node_coord in enumerate(node):
    ax.scatter(node_coord[0], node_coord[1], color=colors[idx], s=20, zorder=5)

# 添加图例说明颜色含义
boundary_patch = mpatches.Patch(color='red', label='Boundary Nodes')
interior_patch = mpatches.Patch(color='blue', label='Interior Nodes')
ax.legend(handles=[boundary_patch, interior_patch])

ax.set_title('Mesh with Boundary and Interior Nodes Highlighted')
ax.set_xlim(-3e-3,3e-3)
ax.set_ylim(-3e-3,3e-3)
plt.show()

