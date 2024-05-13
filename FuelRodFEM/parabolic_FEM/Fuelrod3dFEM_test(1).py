import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.sparse.linalg import spsolve
from fealpy.mesh import TetrahedronMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import DiffusionIntegrator
from fealpy.fem import BilinearForm
from fealpy.fem import ScalarMassIntegrator
from fealpy.fem import LinearForm
from fealpy.fem import ScalarSourceIntegrator

from fealpy.mesh import TetrahedronMesh
import matplotlib.pyplot as plt
import numpy as np
from fealpy.fem.dirichlet_bc import DirichletBC
from scipy.sparse.linalg import gmres

def from_fuel_rod_gmsh(R1,R2,L,w,h,l,p,meshtype='segmented'):
    """
    Generate a tetrahedron mesh for a fuel-rod region by gmsh

    @param R1 The radius of semicircles
    @param R2 The radius of quarter circles
    @param L The length of straight segments
    @param w The thickness of caldding
    @param h Parameter controlling mesh density
    @param l The length of the fuel-rod
    @param p The pitch of the fuel-rod
    @return TetrahedronMesh instance
    """
    import gmsh
    import numpy as np
    import math
    from fealpy.mesh import TetrahedronMesh
    gmsh.initialize()
    gmsh.model.add("fuel_rod_3D")

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
    # 生成内圈线环  
    factory.addCurveLoop(line_list_in,34)

    # 内圈面
    factory.addPlaneSurface([34],35)
    # 包壳截面
    factory.addPlaneSurface([17, 34],36)

    factory.synchronize()

    N = math.ceil((2*l)/p)

    angle = ((2*l)/p* math.pi) / N
    #angle = 0
    #nsection = math.ceil(l/(N* np.sqrt(6)/3 * h))
    nsection = math.ceil(l/(N* 0.4 * h))
    #nsection = math.ceil(l/(N* 2/3 * h))
    #nsection = math.ceil(l/(N* h))
    ov1 = [[0,35]]
    ov2 = [[0,36]]
    if meshtype == 'segmented':
        for i in range(N):
            ov1 = factory.twist([(2,ov1[0][1])],0,0,0,0,0,l/N,0,0,1,angle,[nsection],[],False)
            ov2 = factory.twist([(2,ov2[0][1])],0,0,0,0,0,l/N,0,0,1,angle,[nsection],[],False)
    elif meshtype == 'unsegmented':
        for i in range(N):
            ov1 = factory.twist([(2,ov1[0][1])],0,0,0,0,0,l/N,0,0,1,angle)
            ov2 = factory.twist([(2,ov2[0][1])],0,0,0,0,0,l/N,0,0,1,angle)

    factory.synchronize()
    # 生成网格
    gmsh.model.mesh.generate(3)
    #gmsh.fltk.run()
    # 获取节点信息
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node = np.array(node_coords, dtype=np.float64).reshape(-1, 3)

    #节点的编号映射
    nodetags_map = dict({j:i for i,j in enumerate(node_tags)})

    # 获取四面体单元信息
    tetrahedron_type = 4  
    tetrahedron_tags, tetrahedron_connectivity = gmsh.model.mesh.getElementsByType(tetrahedron_type)
    evid = np.array([nodetags_map[j] for j in tetrahedron_connectivity])
    cell = evid.reshape((tetrahedron_tags.shape[-1],-1))

    print(f"Number of nodes: {node.shape[0]}")
    print(f"Number of cells: {cell.shape[0]}")

    NN = node.shape[0]
    tag2nidx = np.zeros(2*NN,dtype=np.int_)
    tag2nidx[node_tags] = np.arange(NN)

    # 二维面片
    dimtags2 = gmsh.model.getEntities(2)
    # 删去前后和中间插入的面片，剩下两边包裹的
    del dimtags2[0],dimtags2[0],dimtags2[16:len(dimtags2):17]


    # 内边界面片集合
    in_dimtags2 = sum([dimtags2[i:i+16] for i in range(0,len(dimtags2),32)],[])
    # 存储共享边界点
    cntags = []
    for dim,tag in in_dimtags2:
        idx = gmsh.model.mesh.get_elements(dim,tag)[2][0]
        cntags.extend(idx)
    cntags = np.unique(cntags)
    cnidx = tag2nidx[cntags]

    # 外边界面片集合
    out_dimtags2 = sum([dimtags2[i:i+16] for i in range(16,len(dimtags2),32)],[])
    # 存储外边界点
    bdntags = []
    for dim,tag in out_dimtags2:
        idx = gmsh.model.mesh.get_elements(dim,tag)[2][0]
        bdntags.extend(idx)
    bdntags = np.unique(bdntags)
    bdnidx = tag2nidx[bdntags]

    dimtags3 = gmsh.model.getEntities(3)

    # 存储燃料节点编号
    innidx = []
    # 存储包壳节点编号
    canidx = []
    for dim,tag in dimtags3:
        idx = gmsh.model.mesh.get_elements(dim,tag)[2][0]     
        idx = tag2nidx[idx]

        if tag%2 == 1:
            innidx.extend(idx)
        else:
            canidx.extend(idx)
    innidx = np.unique(innidx)
    canidx = np.unique(canidx)

    gmsh.finalize()
    return TetrahedronMesh(node,cell),cnidx,bdnidx,innidx,canidx

mm = 1
#包壳厚度
w = 0.15 * mm
#半圆半径
R1 = 0.5 * mm
#四分之一圆半径
R2 = 1.0 * mm
#连接处直线段
L = 0.575 * mm
#内部单元大小
h = 0.3 * mm
#棒长
l = 20 * mm
#螺距
p = 40 * mm


""""
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
h = 0.8 * mm
#棒长
l = 4 * mm
#螺距
p = 10 * mm

"""
class ParabolicData:
    
    

    def duration(self):
        """
        @brief 时间区间
        """
        return [0, 10]
    
    def source(self):
        
        return 0
    
    def dirichlet(self, p):
        """
        @brief 返回 Dirichlet 边界上的给定点的位移
        @param[in] p 一个表示空间点坐标的数组
        @return 返回位移值，这里返回常数向量 [0.0, 0.0,0.0]
        """
        return  500



mesh,cnidx,bdnidx,innidx,canidx = from_fuel_rod_gmsh(R1,R2,L,w,h,l,p,meshtype='segmented')
print(len(cnidx))
print(len(bdnidx))
print(len(innidx))
print(len(canidx))
pde=ParabolicData()
source=pde.source()
node = mesh.node
NC=mesh.number_of_cells
NN=mesh.number_of_nodes
print("单元个数：",NC)
print("节点个数",NN)
print("反转单元个数:",np.sum(mesh.entity_measure("cell")<=0))

def show_quality(self, axes, qtype=None, quality=None):
    #@brief 显示网格质量分布的分布直方图
    minq = np.min(1/quality)
    maxq = np.max(1/quality)
    meanq = np.mean(1/quality)
    hist, bins = np.histogram(1/quality, bins=50, range=(0, 1))
    center = (bins[:-1] + bins[1:]) / 2
    axes.bar(center, hist, align='center', width=0.02)
    axes.set_xlim(0, 1)
    axes.annotate('Min quality: {:.6}'.format(minq), xy=(0, 0), xytext=(0.1, 0.5),
                    textcoords='axes fraction',
                    horizontalalignment='left', verticalalignment='top')
    axes.annotate('Max quality: {:.6}'.format(maxq), xy=(0, 0), xytext=(0.1, 0.45),
                    textcoords='axes fraction',
                    horizontalalignment='left', verticalalignment='top')
    axes.annotate('Average quality: {:.6}'.format(meanq), xy=(0, 0), xytext=(0.1, 0.40),
                    textcoords='axes fraction',
                    horizontalalignment='left', verticalalignment='top')
    return minq, maxq, meanq
fig,axes1= plt.subplots()
quality = mesh.cell_quality()
show_quality(mesh,axes1,quality=quality)
plt.show()

isBdNode = mesh.ds.boundary_node_flag()
print(isBdNode.shape)


# 时间离散
duration = pde.duration()
nt = 64
tau = (duration[1] - duration[0])/nt 

####全局矩阵组装####
space=LagrangeFESpace(mesh, p=1)


####内部的矩阵组装####
# 基函数
space = LagrangeFESpace(mesh, p=1)
GD=space.geo_dimension()


phi=space.basis
print("phi",phi)
#载荷向量
bform3=LinearForm(space)
bform3.add_domain_integrator(ScalarSourceIntegrator(source,q=3))
F=bform3.assembly()
print(F)

#组装刚度矩阵
alpha_caldding=0.4
alpha_inner=0.8
alpha=np.zeros_like(F)
alpha[canidx]+=alpha_caldding
alpha[innidx]+=alpha_inner
print(alpha)
bform = BilinearForm(space)
bform.add_domain_integrator(DiffusionIntegrator(q=3))
K= bform.assembly()

#组装质量矩阵
bform2=BilinearForm(space)
bform2.add_domain_integrator(ScalarMassIntegrator(q=3))
M=bform2.assembly()
print(M)


p_0=np.zeros_like(F)
p_0+=300
alpha_caldding=0.4
alpha_inner=0.8


import os
output = './result_fuelrod_cnidx3d'
filename = 'temp'
# Check if the directory exists, if not, create it
if not os.path.exists(output):
    os.makedirs(output)

for n in range(nt):
    """
    if n == 0:
        p_0 = p_0[:]
    else:
        p_0 = p_1[:]
    """
    t = duration[0] + n*tau
    print("M",M)
    print("K",K)
    A= M+alpha_caldding*K*tau
    print("A",A)
    print("p_0",p_0)
    b = M@p_0 + tau*F
    print("b",b)
    bc = DirichletBC(space = space, gD = pde.dirichlet) 
    A,b = bc.apply(A,b)
    p_0[cnidx] = pde.dirichlet(node[cnidx])
    #p_0=spsolve(A,b)
    x0 = np.zeros_like(b)  # 初始猜测向量，通常是零向量
    tolerance = 1e-16  # 解的容差
    maxiter = 1000  # 最大迭代次数

    # 调用 gmres
    p_0, exitCode = gmres(A, b, x0=x0, rtol=tolerance, maxiter=maxiter)
    
    mesh.nodedata['temp'] = p_0.flatten('F')
    name = os.path.join(output, f'{filename}_{n:010}.vtu')
    mesh.to_vtk(fname=name)

print(p_0)
print(p_0.shape)
