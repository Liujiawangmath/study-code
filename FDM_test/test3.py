import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from fealpy.mesh import UniformMesh1d
import matplotlib.pyplot as plt # 绘图
class EXPX2PDEData:
    """
    -u''(x)+2 u = 4*x**4*exp(-x**2) - 16*x**2*exp(-x**2) + 6*exp(-x**2)  
       u(-1) = 0, u(1) = 0.
    exact solution：
       u(x) = exp(-x**2)*(1-x**2).
    """
    def domain(self):
       
        return [-1, 1]
 
    def solution(self, p: np.ndarray):
        
        return np.exp(-p**2)*(1-p**2)
    
    def source(self, p: np.ndarray):
        
        return 4*p**4*np.exp(-p**2) - 15*p**2*np.exp(-p**2) + 5*np.exp(-p**2) 
    

    def gradient(self, p: np.ndarray):
       
        return -2*p*(1 - p**2)*np.exp(-p**2) - 2*p*np.exp(-p**2) 

   
    def dirichlet(self, p: np.ndarray):
        """
        @brief: Dirichlet BC
        """
        return self.solution(p)

pde = EXPX2PDEData()
domain = pde.domain()
nx = 20 # 网格节点个数
hx = (domain[1] - domain[0])/nx #步长
mesh = UniformMesh1d([0, nx], h = hx, origin = domain[0]) #剖分网格
NN = mesh.number_of_nodes() # 节点数
maxit = 4
uh = mesh.function()
et = ['$|| u - u_h||_{\infty}$', '$|| u - u_h||_{0}$', '$|| u - u_h ||_{1}$']
eu = np.zeros(len(et), dtype=np.float64) 
eu[0], eu[1], eu[2] = mesh.error(pde.solution, uh)
et = np.array(et)
em = np.zeros((len(et), maxit), dtype=np.float64)
egradm = np.zeros((len(et), maxit), dtype=np.float64) 
x= np.linspace(-1,1,100)
y = pde.solution(x)
fig = plt.figure()
ax = fig.gca()
ax.plot(x,y)
plt.show()

node = mesh.entity('node')

for i in range(maxit):
    A = mesh.laplace_operator()
    NN = mesh.number_of_nodes()
    val = np.broadcast_to(1, (NN, ))
    k = np.arange(NN)
    I = k[:]
    A += csr_matrix((val, (I, I)), shape=(NN, NN), dtype=mesh.ftype)
    uh = mesh.function() 
    f = mesh.interpolate(pde.source, 'node')
    A, f = mesh.apply_dirichlet_bc(gD=pde.dirichlet, A=A, f=f)
    uh[:] = spsolve(A, f)
    node = mesh.entity('node').reshape(-1)
    uI = pde.solution(node)
    e = np.abs(uh-uI)

    fig = plt.figure()
    axes = fig.gca()
    #mesh.show_function(axes, uh)
    mesh.show_function(axes, e)   
    plt.show()

    em[0, i], em[1, i], em[2, i] = mesh.error(pde.solution, uh)

    if i < maxit:
        mesh.uniform_refine()

print("em:\n", em)
print("em_ratio:\n", em[:, 0:-1]/em[:, 1:])