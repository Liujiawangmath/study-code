import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh
def from_fuel_rod_gmsh(R1,R2,L,w,h,meshtype='normal'):
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
    # 内圈面
    factory.addPlaneSurface([34],35)
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

    print(f"Number of nodes: {node.shape[0]}")
    print(f"Number of cells: {cell.shape[0]}")

    NN = len(node)
    isValidNode = np.zeros(NN, dtype=np.bool_)
    isValidNode[cell] = True
    node = node[isValidNode]
    idxMap = np.zeros(NN, dtype=cell.dtype)
    idxMap[isValidNode] = range(isValidNode.sum())
    cell = idxMap[cell]

    tag2nidx = np.zeros(2*NN,dtype=np.int_)
    tag2nidx[node_tags] = np.arange(NN)
    dimtags1 = gmsh.model.getEntities(1)
    # 共享节点编号
    cnidx = []
    # 边界节点编号
    bdnidx = []
    for dim, tag in dimtags1:
        ntags = gmsh.model.mesh.get_elements(dim,tag)[2][0]
        idx = tag2nidx[ntags]
        if tag < 17:
            bdnidx.extend(idx)
        else:
            cnidx.extend(idx)
    cnidx = np.unique(cnidx)
    bdnidx = np.unique(bdnidx)
    # 内部节点编号和包壳节点编号
    inntags = gmsh.model.mesh.get_elements(2,35)[2][0]
    cantags = gmsh.model.mesh.get_elements(2,36)[2][0]
    innidx = np.unique(tag2nidx[inntags])
    canidx = np.unique(tag2nidx[cantags])

    gmsh.fltk.run()
    gmsh.finalize()
    return TriangleMesh(node,cell),cnidx,bdnidx,innidx,canidx


mm = 1e-03
#包壳厚度
w = 0.15 * mm
#半圆半径
R1 = 0.5 * mm
#四分之一圆半径
R2 = 1 * mm
#连接处直线段
L = 0.575 * mm
#内部单元大小
h = 0.2 * mm
mesh,cnidx,bdnidx,innidx,canidx = from_fuel_rod_gmsh(R1,R2,L,w,h,meshtype='normal')
print(len(cnidx))
print(len(bdnidx))
print(len(innidx))
print(len(canidx))
#print(mesh.entity('cell'))
"""
def plot_delaunay(mesh):
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        plt.triplot(node[:, 0], node[:, 1], cell)
        #plt.plot(node[:, 0], node[:, 1],'ro')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
plot_delaunay(mesh)
"""