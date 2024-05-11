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

mesh=from_fuel_rod_gmsh(R1,R2,L,w,h,meshtype='normal',square='all')
mesh_inner = from_fuel_rod_gmsh(R1,R2,L,w,h,meshtype='normal',square='inner')
mesh_caldding=from_fuel_rod_gmsh(R1,R2,L,w,h,meshtype='normal',square='caldding')

pde=ParabolicData()
source=pde.source()

node = mesh.node
node_caldding=mesh_caldding.node
node_inner=mesh_inner.node

isBdNode = mesh_inner.ds.boundary_node_flag()
print(isBdNode.shape)
isBdNode_caldding = mesh.ds.boundary_node_flag()
print(isBdNode_caldding.shape)

# 初始化共享节点列表和两个拓扑关系字典
BdNode = []
BdNode1 = np.zeros(len(node_caldding), dtype=bool)
BdNode2 = np.zeros(len(node_inner), dtype=bool)

# 遍历第一组节点
for i, node1 in enumerate(node_caldding):
    # 遍历第二组节点
    for j, node2 in enumerate(node_inner):
        # 如果节点坐标完全相同，则是共享节点
        if np.array_equal(node1, node2):
            # 将共享节点坐标添加到BdNode中
            BdNode.append(node1)
            # 记录共享节点在第一组节点中的编号索引
            # 将对应位置的布尔值设置为True
            BdNode1[i] = True
            BdNode2[j] = True



isBdNode_caldding=mesh_caldding.ds.boundary_node_flag()
isBdNode_caldding =np.logical_xor(isBdNode_caldding, BdNode1)

import matplotlib.pyplot as plt
fig = plt.figure()
axes = fig.gca()
mesh_inner.add_plot(axes)
mesh_inner.find_cell(axes, showindex=True, color='k', marker='s', markersize=2, fontsize=8, fontcolor='k')
mesh_inner.find_node(axes, showindex=True, color='r', marker='o', markersize=2, fontsize=8, fontcolor='r')
#plt.show()

# 时间离散
duration = pde.duration()
nt = 640
tau = (duration[1] - duration[0])/nt 

####全局矩阵组装####
space=LagrangeFESpace(mesh, p=1)
bform3=LinearForm(space)
bform3.add_domain_integrator(ScalarSourceIntegrator(source,q=3))
F=bform3.assembly()

####内部的矩阵组装####
# 基函数
space_inner = LagrangeFESpace(mesh_inner, p=1)
GD=space_inner.geo_dimension()
#组装刚度矩阵
bform = BilinearForm(space_inner)
bform.add_domain_integrator(DiffusionIntegrator(q=3))
K_inner = bform.assembly()
#组装质量矩阵
bform2=BilinearForm(space_inner)
bform2.add_domain_integrator(ScalarMassIntegrator(q=3))
M_inner=bform2.assembly()
#载荷向量
bform3=LinearForm(space_inner)
bform3.add_domain_integrator(ScalarSourceIntegrator(source,q=3))
F_inner=bform3.assembly()


####包壳的矩阵组装####
# 基函数
space_caldding = LagrangeFESpace(mesh_caldding, p=1)
GD=space_caldding.geo_dimension()
#组装刚度矩阵
bform = BilinearForm(space_caldding)
bform.add_domain_integrator(DiffusionIntegrator(q=3))
K_caldding = bform.assembly()
#组装质量矩阵
bform2=BilinearForm(space_caldding)
bform2.add_domain_integrator(ScalarMassIntegrator(q=3))
M_caldding=bform2.assembly()
#载荷向量
bform3=LinearForm(space_caldding)
bform3.add_domain_integrator(ScalarSourceIntegrator(source,q=3))
F_caldding=bform3.assembly()



p=np.zeros_like(F)
p+=300
alpha_caldding=4e-3
alpha_inner=8e-3
p_caldding = np.zeros_like(F_caldding)
p_inner=np.zeros_like(F_inner)

import os
output = './result_fuelrod'
filename = 'temp'
# Check if the directory exists, if not, create it
if not os.path.exists(output):
    os.makedirs(output)

for n in range(nt):
    t = duration[0] + n*tau
    A_caldding = M_caldding + alpha_caldding*K_caldding*tau
    b_caldding = M_caldding @ p_caldding + tau*F_caldding
    p_caldding=spsolve(A_caldding,b_caldding)

    # 全局Dirichlet边界条件
     # 遍历共享节点
    #for i, shared_node in enumerate(BdNode):
        # 找到共享节点在第一组节点中的索引
        #index_caldding = np.where((node_caldding == shared_node).all(axis=1))[0][0]
        # 找到共享节点在第二组节点中的索引
        #index_inner = np.where((node_inner == shared_node).all(axis=1))[0][0]
        # 将第一组节点上的温度赋值给第二组节点
        #p_inner[index_inner] = p_caldding[index_caldding]
        
    index_caldding = np.where((node_caldding[:, None] == BdNode).all(axis=2))[1]
    index_inner = np.where((node_inner[:, None] == BdNode).all(axis=2))[1]
    p_inner[index_inner] = p_caldding[index_caldding]

    
    A_inner = M_inner + alpha_inner*K_inner*tau
    b_inner = M_inner @ p_inner + tau*F_inner
    p_inner=spsolve(A_inner,b_inner)
   
    p_caldding[BdNode1] = pde.dirichlet(node_caldding)

    # 将 p_caldding 中的值填入 p 中对应的位置
    #for i, l in enumerate(node_caldding):
        #global_index = np.where((node == l).all(axis=1))[0][0]
        #p[global_index] = p_caldding[i]

    # 将 p_inner 中的值填入 p 中对应的位置
    #for i, l in enumerate(node_inner):
        #global_index = np.where((node == l).all(axis=1))[0][0]
        #p[global_index] = p_inner[i]
    # 将节点坐标转换成对应的全局索引
    global_indices_caldding = np.where(np.all(node[:, None] == node_caldding, axis=2))[0]
    global_indices_inner = np.where(np.all(node[:, None] == node_inner, axis=2))[0]

# 将 p_caldding 中的值填入 p 中对应的位置
    p[global_indices_caldding] = p_caldding

# 将 p_inner 中的值填入 p 中对应的位置
    p[global_indices_inner] = p_inner

    
    mesh.nodedata['temp'] = p.flatten('F')
    name = os.path.join(output, f'{filename}_{n:010}.vtu')
    mesh.to_vtk(fname=name)

print(p)
print(p.shape)