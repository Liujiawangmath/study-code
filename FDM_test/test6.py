import numpy as np
from sympy import *
from fealpy.mesh import UniformMesh2d
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags, csr_matrix
class expsinsinData_2d:
    def __init__(self,u:str,x:str='x',y:str='y',t:str='t',k=1.0,D=[0,1,0,1],T=[0,1]):
        self.u = lambdify([x,y,t],simplify(u))
        self.f = lambdify([x,y,t],diff(u,t,1)-k*diff(u,x,2)-k*diff(u,y,2))
        self.dudx = lambdify([x,y,t],diff(u,x,1))
        self.dudy = lambdify([x,y,t],diff(u,y,1))
        self._domain = D
        self._duration = T
    def domain(self):
        return self._domain
    def duration(self):
        return self._duration
    def solution(self,p,t):
        x = p[...,0]
        y = p[...,1]
        return self.u(x,y,t)
    
    def init_solution(self,p):
        return self.solution(p,0)
    
    def source(self,p,t):
        x = p[...,0]
        y = p[...,1]
        return self.f(x,y,t)
    def gradient(self,p,t):
        x = p[...,0]
        y = p[...,1]
        val = np.zeros((len(x),2))
        val[...,0] = self.dudx(x,y,t)
        val[...,1] = self.dudy(x,y,t)
        return val
    def dirichlet(self,p,t):
        return self.solution(p,t)
    
pde = expsinsinData_2d('exp(-t)*sin(pi*x)*sin(pi*y)')
domain = pde.domain()
nx = 20
ny = 20
hx = (domain[1]-domain[0])/nx
hy = (domain[3]-domain[2])/ny
mesh = UniformMesh2d([0,nx,0,ny],h = (hx,hy),origin=(domain[0],domain[2]))

duration = pde.duration()
nt = 1000
tau = (duration[1]-duration[0])/nt

u0 = mesh.interpolate(pde.init_solution,'node')

def advance_crank_nicholson(n):

    t = duration[0] + n*tau
    if n == 0:
        return u0, t
    else:
        A, B = mesh.parabolic_operator_crank_nicholson(tau)
        source = lambda p: pde.source(p, t )
        f = mesh.interpolate(source, intertype='node') # f.shape = (nx+1,ny+1)
        f *= tau
        f.flat[:] += B@u0.flat[:]
         
        gD = lambda p: pde.dirichlet(p, t )
        A, f = mesh.apply_dirichlet_bc(gD, A, f)
        u0.flat = spsolve(A, f)

        solution = lambda p: pde.solution(p, t )
        e = mesh.error(solution, u0, errortype='max')
        
        print(f"the max error is {e}")
        
        return u0, t
    
"""
e_m = []
for j in range(4):
    for i in range(2):
        if i >= 1:
            _,_,e = advance_crank_nicholson(i)
            e_m.append(e)
    if j < 4:
        mesh.uniform_refine()
        uh0 = mesh.interpolate(pde.init_solution,'node')
e_m = np.array(e_m,dtype=np.float64)
print(e_m)
e_ratio = e_m[:-1]/e_m[1:]
print(e_ratio)
"""
"""
for k in range(nt+1):
    uh0 ,t  = advance_crank_nicholson(k)
    node = mesh.entity('node')
    e0 = pde.solution(node,t).reshape(len(uh0),-1) - uh0 
    if t == 1:
        fig1 = plt.figure(1)
        axes1 = fig1.add_subplot(111 , projection='3d')
        mesh.show_function(axes1, uh0)
        plt.show()

        fig2 = plt.figure(2)
        axes2 = fig2.add_subplot(111 , projection='3d')
        mesh.show_function(axes2, e0)
        plt.show()

        x = np.linspace(0, 1, 101)
        y = np.linspace(0, 1, 101)
        X, Y = np.meshgrid(x, y)
        p = np.array([X, Y]).T
        Z = pde.solution(p,1)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='jet')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
"""
#fig,axes = plt.subplots()
#box = [0,1,0,1,-1,1]
#axes = fig.add_subplot(111, projection='3d')
#mesh.show_animation(fig, axes, box, advance_crank_nicholson, fname='cn.mp4', plot_type='surface', frames=nt + 1)
#plt.show()

def parabolic_operator_richardson(mesh,tau):

    rx = tau / mesh.h[0]**2
    ry = tau / mesh.h[1]**2

    NN = mesh.number_of_nodes()
    n0 = mesh.nx + 1
    n1 = mesh.ny + 1
    K = np.arange(NN).reshape(n0, n1)

    A = diags([ -4*rx - 4*ry], [0], shape=(NN, NN), format='csr')

    val = np.broadcast_to(2 * rx, (NN - n1,))
    I = K[1:, :].flat
    J = K[0:-1, :].flat
    A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=mesh.ftype)
    A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=mesh.ftype)

    val = np.broadcast_to(2 * ry, (NN - n0,))
    I = K[:, 1:].flat
    J = K[:, 0:-1].flat
    A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=mesh.ftype)
    A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=mesh.ftype)

    B = diags([1], [0], shape=(NN, NN), format='csr')
    return A , B

def Du_Fort_Frankel(mesh,tau):
    rx = tau / mesh.h[0]**2
    ry = tau / mesh.h[1]**2

    NN = mesh.number_of_nodes()
    n0 = mesh.nx + 1
    n1 = mesh.ny + 1
    K = np.arange(NN).reshape(n0, n1)

    A = diags([0], [0], shape=(NN, NN), format='csr')

    val = np.broadcast_to(2 * rx, (NN - n1,))
    I = K[1:, :].flat
    J = K[0:-1, :].flat
    A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=mesh.ftype)
    A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=mesh.ftype)

    val = np.broadcast_to(2 * ry, (NN - n0,))
    I = K[:, 1:].flat
    J = K[:, 0:-1].flat
    A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=mesh.ftype)
    A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=mesh.ftype)

    B = diags([1 - 2*rx - 2*ry], [0], shape=(NN, NN), format='csr')
    C = diags([1 + 2*rx + 2*ry], [0], shape=(NN, NN), format='csr')
    return A , B ,C


A, B = parabolic_operator_richardson(mesh,tau)
def advance_richardson(n):
    t = duration[0] + n*tau
    if n == 0:
        return mesh.interpolate(pde.init_solution,'node'), t
    elif n == 1:
        A1, B1 = mesh.parabolic_operator_crank_nicholson(tau)
        source = lambda p: pde.source(p, t )
        f = mesh.interpolate(source, intertype='node') # f.shape = (nx+1,ny+1)
        f *= tau
        f.flat[:] += B1@uh0.flat[:]
         
        gD = lambda p: pde.dirichlet(p, t )
        A1, f = mesh.apply_dirichlet_bc(gD, A1, f)
        uh1.flat = spsolve(A1, f)
        return uh1, t
    else:
        source = lambda p: pde.source(p, t+tau)
        f = mesh.interpolate(source, intertype='node') # f.shape = (nx+1,ny+1)
        f *= 2*tau
 
        f.flat[:] += B@uh0.flat[:]
        uh2.flat = A@uh1.flat + f.flat

        gD = lambda p: pde.dirichlet(p, t+tau )
        mesh.update_dirichlet_bc(gD,uh2)
        uh0[:] = uh1
        uh1[:] = uh2
        solution = lambda p: pde.solution(p, t )
        e = mesh.error(solution, uh2, errortype='max')
        
        print(f"the max error is {e}")
        
        return uh2, t
    
uh0 = mesh.interpolate(pde.init_solution,'node')    
uh1 = mesh.function('node')
uh2 = mesh.function('node')

def advance_DFF(n):
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    elif n == 1:
        
        A1, B1 = mesh.parabolic_operator_crank_nicholson(tau)
        source = lambda p: pde.source(p, t )
        f = mesh.interpolate(source, intertype='node') # f.shape = (nx+1,ny+1)
        f *= tau
        f.flat[:] += B1@uh0.flat[:]
         
        gD = lambda p: pde.dirichlet(p, t )
        A1, f = mesh.apply_dirichlet_bc(gD, A1, f)
        uh1.flat = spsolve(A1, f)
        
        return uh1,t
    else:
        A,B,C = Du_Fort_Frankel(mesh,tau)

        source0 = lambda p: pde.source(p, t-tau )
        source1 = lambda p: pde.source(p, t )
        source2 = lambda p: pde.source(p, t+tau )

        f0 = mesh.interpolate(source0, intertype='node') # f.shape = (nx+1,ny+1)
        f1 = mesh.interpolate(source1, intertype='node')
        f2 = mesh.interpolate(source2, intertype='node')

        f = (tau/2)*(f0 + 2*f1 +f2)

        f.flat[:] += (A@uh1.flat[:] + B@uh0.flat[:])
        gD = lambda p: pde.dirichlet(p, t+tau )
        C, f = mesh.apply_dirichlet_bc(gD, C, f)
        uh2.flat = spsolve(C, f)
        uh0[:] = uh1
        uh1[:] = uh2
        solution = lambda p: pde.solution(p, t+tau )
        e = mesh.error(solution, uh1, errortype='max')
        
        print(f"the max error is {e}")
        
        return uh1, t

# for i in range(3):
#     advance_DFF(i)

for k in range(nt+1):
    u,t  = advance_richardson(k)
    #uh2 = uh2
    #u,t  = advance_DFF(k)
    node = mesh.entity('node')
    e0 = pde.solution(node,t).reshape(len(u),-1) - u
    #e1 = pde.solution(node,t).reshape(len(u),-1) - u
    if t == 0.1:
        fig1 = plt.figure(1)
        axes1 = fig1.add_subplot(111 , projection='3d')
        mesh.show_function(axes1, u)

        fig2 = plt.figure(2)
        axes2 = fig2.add_subplot(111 , projection='3d')
        mesh.show_function(axes2, e0)


        #fig3 = plt.figure(3)
        #axes3 = fig3.add_subplot(111 , projection='3d')
        #mesh.show_function(axes3, u)

        #fig4 = plt.figure(4)
        #axes4 = fig4.add_subplot(111 , projection='3d')
        #mesh.show_function(axes4, e1)


        x = np.linspace(0, 1, 101)
        y = np.linspace(0, 1, 101)
        X, Y = np.meshgrid(x, y)
        p = np.array([X, Y]).T
        Z = pde.solution(p,1)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='jet')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
