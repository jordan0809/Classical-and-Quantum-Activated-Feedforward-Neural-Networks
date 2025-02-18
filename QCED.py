import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy.spatial.distance import pdist, squareform
import time 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Construct 1-qubit Pauli operator
def single_pauli(q,j,axis):  #q: number of qubits, target jth qubit, axis of the Pauli matrix (x->0, y->1, z->2)
    product = [qeye(2)]*q
    paulis =[sigmax(),sigmay(),sigmaz()]
    product[j] = paulis[axis]
    return tensor(product)

# Construct 1-qubit number operator 
def nz(q,j):  #q: number of qubits, target jth qubit
    product = [qeye(2)]*q
    product[j] = (1+sigmaz())/2
    return tensor(product)

# Construct 2-qubit number operator 
def nznz(q,j,k):  #q: number of qubits, target jth and kth qubit
    product = [qeye(2)]*q
    product[j] = (1+sigmaz())/2
    product[k] = (1+sigmaz())/2
    return tensor(product)

# Calculate the Rydberg interaction matrix based on the given coordinates
def ryd_int(coords,C=1):
    interactions = squareform(C/pdist(coords)**6)
    return interactions

# Quantum simulation of the Rydberg Hamiltonian
# The Rydberg Hamiltonian is composed of one global Ω(t), q local Δj(t), and q(q-1)/2 Rydberg interactions Vjk
# Ω(t) is parametrized by the asymptotic values of the basis step functions, while Δj(t) by the initial point Δj(0) and the slope sj
# Ω(t), Δj(0),and sj each has q parameters (in total 3q parameters)
def simulation(q,T,omegas,delta_0,delta_s,V,noise=0): 
    """
    T:evolution time 
    omegas: Ω(t) parameters
    delta_0: Δj(0)
    delta_s: sj
    V: Rydberg interaction matrix
    noise: maximum random noise ratio (relative to the laser parameters)

    """
    omegas = omegas*(1+np.random.uniform(-noise,noise))
    delta_0 = delta_0*(1+np.random.uniform(-noise,noise))
    delta_s = delta_s*(1+np.random.uniform(-noise,noise))

    #N = len(omegas)+2  #include boundary values
    #oparams = [0]+list(omegas)+[0]  #boundary conditions: Ω(t0) = Ω(tN) = 0
    N = len(omegas)  #dont consider boundary conditions
    oparams = list(omegas)
    
    delta_t = T/N  #duration of each step

    w = [oparams[i] if i==0 else oparams[i]-oparams[i-1] for i in range(N)] #coefficients of the basis Sigmoid functions

    try: 
        #Cython
        import cython  #if Cython is installed, string-based Hamiltonian is used in QuTiP simulation
        
        omega_t = ""
        for i in range(1,N): 
            omega_t += f"{w[i]}*1/(exp(-1000*(t-{delta_t}*{i})+10)+1)+"

        omega_t = omega_t[:-1]  #remove the "+" sign in the end

        delta_t = []
        for j in range(q):
            d = f"{delta_0[j]}+{delta_s[j]}*t"
            delta_t.append(d)

    except ImportError:
        #Without Cython (function-based Hamiltonian is used)
        def omega_t(t,args):
            y = 0
            for i in range(N):
                y += w[i]*1/(np.exp(-1000*(t-i*delta_t)+10)+1)
            return y

        class local_delta:
            def __init__(self,j):
                self.j = j

            def evo(self):
                func = lambda t,args: delta_0[self.j]+delta_s[self.j]*t
                return func

        delta_t = [local_delta(j).evo() for j in range(q)]

    
    Vjk = 0
    for j in range(q):
        for k in range(j+1,q):
            Vjk += V[j,k]*nznz(q,j,k)
        
    #time-independent part
    H_t = [Vjk]

    #time-dependent part
    for j in range(q):
        H_t.append([0.5*single_pauli(q,j,0),omega_t])
        H_t.append([-nz(q,j),delta_t[j]])
    
        
    time_list = np.linspace(0,T,int(T)) 

    #intial state = ground state = |1> 
    init = tensor([basis(2,1) for i in range(q)])

    result_t = mesolve(H_t, init, time_list,[], [])

    #Measure in the z-basis of each qubit
    measurement = [single_pauli(q,j,2) for j in range(q)]

    #Expectation value measured in each qubit
    expval_list = np.array([expect(measurement[j],result_t.states[-1]) for j in range(q)])

    return expval_list

# Finite-difference method for gradient computation
def finite_diff(q,T,omegas,delta_0,delta_s,V,noise):

    diff = 0.0001
    shift = diff*np.identity(q)

    omegas_grad = []
    delta_0_grad = []
    delta_s_grad = []

    for m in range(q):  #m: index of the parameter
        plus = omegas+shift[m,:]
        minus = omegas-shift[m,:]
        omegas_grad.append((simulation(q,T,plus,delta_0,delta_s,V,noise)-simulation(q,T,minus,delta_0,delta_s,V,noise))/(2*diff))
        plus = delta_0+shift[m,:]
        minus = delta_0-shift[m,:]
        delta_0_grad.append((simulation(q,T,omegas,plus,delta_s,V,noise)-simulation(q,T,omegas,minus,delta_s,V,noise))/(2*diff))
        plus = delta_s+shift[m,:]
        minus = delta_s-shift[m,:]
        delta_s_grad.append((simulation(q,T,omegas,delta_0,plus,V,noise)-simulation(q,T,omegas,delta_0,minus,V,noise))/(2*diff))
    
    ograd = np.array(omegas_grad).transpose()   #jth row: gradients w.r.t. the expectation value of the jth qubit
    dgrad_0 = np.array(delta_0_grad).transpose()
    dgrad_s = np.array(delta_s_grad).transpose()

    gradient_list = [[ograd[j,:],dgrad_0[j,:],dgrad_s[j,:]] for j in range(q)] #jth item: gradients of the expectation value of the jth qubit

    return gradient_list


# Define the Encoder class
# Number of layers in the encoder can be specified in the argument
class Encoder(nn.Module):
    def __init__(self,q,Q_size,layers,omega_max,delta_max,T):
        """
        Q_size: number of variables in the QUBO instance 
        layers: number of layers
        omega_max: maximum allowed Ω
        delta_max: maximum allowed Δ
        
        """
        super(Encoder,self).__init__()
        self.q = q
        self.omega_max = omega_max
        self.delta_max = delta_max
        self.T = T
        self.input_size =int(Q_size*(Q_size+1)/2)  #dimension of the input QUBO data
        self.layers = layers
        self.hidden = nn.ModuleList()
        self.hidden.append(nn.Linear(self.input_size,self.input_size+3*self.q))
        for i in range(1,layers):
            if i<layers-1:
                self.hidden.append(nn.Linear(3*self.q+self.input_size, 3*self.q+self.input_size))
            else:
                self.hidden.append(nn.Linear(3*self.q+self.input_size,3*self.q))
        
        self.sig = nn.Sigmoid()  #activation function of the last encoding layer
        self.relu = nn.ReLU()
        
    def forward(self,x):
        out = x
        for i in range(self.layers):
            if i < self.layers-1:
                func = self.hidden[i]
                out = func(out)
                out = self.relu(out)
            else:
                func = self.hidden[i]
                out = func(out)
                out = self.sig(out)
   
        omega_out = (out[:self.q]*self.omega_max).reshape(self.q,1)   #Ω domain [0,omega_max]
        delta_out_0 = (-self.delta_max/2+out[self.q:2*self.q]*self.delta_max).reshape(self.q,1)  #Δ(0) domain [-delta_max/2,delta_max/2]
        delta_out_s = (-self.delta_max/(2*self.T)+out[2*self.q:]*self.delta_max/self.T).reshape(self.q,1) #s domain [-delta_max/2T, delta_max/2T]

        return torch.cat((omega_out,delta_out_0,delta_out_s),dim=1)


# Define the quantum simulation as the PyTorch Autograd function
class Quantum(torch.autograd.Function):

    #input is the input tensor from the encoder
    @staticmethod
    def forward(ctx,input,q,T,V,noise):  
        
        ctx.q = q #save q as a parameter of ctx to use it in the backward pass

        #turn the input tensors from the encoder into numpy arrays
        t_omegas = input[:,0]
        omegas= t_omegas.detach().numpy()
        
        t_delta_0 = input[:,1]
        delta_0 = t_delta_0.detach().numpy()

        t_delta_s = input[:,2]
        delta_s = t_delta_s.detach().numpy()

        #calculate the gradients of the laser parameters and save it in ctx for the backward pass
        finite = finite_diff(q,T,omegas,delta_0,delta_s,V,noise)
        ctx.finite = finite

        return torch.tensor(simulation(q,T,omegas,delta_0,delta_s,V,noise),dtype=torch.float32,requires_grad=True)


    #grad_output is the gradient of loss w.r.t. each output expectation value 
    @staticmethod
    def backward(ctx, grad_output):

        finite = ctx.finite
        q = ctx.q

        input_grad = []
        for j in range(q):
            grad_omegas = list(finite[j][0])
            grad_delta_0 = list(finite[j][1])
            grad_delta_s = list(finite[j][2])

            input_grad.append(grad_omegas+grad_delta_0+grad_delta_s)

        #compute the gradient of loss w.r.t to each laser parameter with the chain rule
        input_grad = torch.tensor(input_grad,dtype=torch.float32)   #shape (q,3q)
        grad_output = torch.reshape(grad_output,(1,grad_output.shape[0]))  #shape (1,q)
        loss_input = torch.matmul(grad_output,input_grad)   #shape (1,3q)

        loss_omegas = torch.transpose(loss_input[:,:q],0,1) #shape (q,1)
        loss_delta_0 = torch.transpose(loss_input[:,q:2*q],0,1) #shape (q,1)
        loss_delta_s = torch.transpose(loss_input[:,2*q:],0,1) #shape (q,1)

        return torch.cat((loss_omegas,loss_delta_0,loss_delta_s),dim=1),None,None,None,None  #shape (q,3) (same dimension as input)
        #return None for extra parameters q,T,V,noise since we don't need to calculate the gradient of them 

# Define the Decoder Class
# Number of layers in the decoder can be specified in the argument
class Decoder(nn.Module):
    def __init__(self,q,Q_size,layers):
        super(Decoder,self).__init__()
        self.layers = layers
        self.hidden = nn.ModuleList()
        self.hidden.append(nn.Linear(q,q+Q_size))
        for i in range(1,layers):
            if i<layers-1:
                self.hidden.append(nn.Linear(q+Q_size, q+Q_size))
            else:
                self.hidden.append(nn.Linear(q+Q_size,Q_size))
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh() #activate function of the last decoding layer
        
    def forward(self,x):
        out = x
        for i in range(self.layers):
            if i < self.layers-1:
                func = self.hidden[i]
                out = func(out)
                out = self.relu(out)
            else:
                func = self.hidden[i]
                out = func(out)
                out = 0.5*self.tanh(out)+0.5
        
        return out

# QUBO loss function
def qubo_loss(Q,x):
    off = Q.detach().clone()
    off.fill_diagonal_(0)
    return torch.dot(x,torch.diag(Q))+x.t()@off@x  #linear terms + quadratic terms
 


def QCED_solve(Q: Tensor,q_resource: dict,num_epochs=100,lr=0.01,e_layers=3,d_layers=3,noise=0,Q_sol=None):
    """
    Q: QUBO matrix. Should be of type torch.Tensor 
    
    q_resource: quantum resource of the Rydberg annealer. It should be a dictionary with keys in the order: "q","T","coords","omega_max",and "delta_max"
    
    i.e. q_resource = {"q":4,"T":5,"coords":np.array([[0., 0.],
                                                      [0., 1.],
                                                      [1., 0.],
                                                      [1., 1.]]),
                    "omega_max":5,"delta_max":20}
                    
    where q is the number of qubits, T is the evolution time, coords is the coordinates array of atoms, 
    omega_max and delta_max are the maximum allowed Ω and Δ in the simulation.

    num_epochs: number of training epochs

    lr: learning rate

    e_layer, d_layer: number of layers in the encoder and decoder respectively

    noise: maximum noise ratio relative to the laser parameters
                        
    Q_sol: cost value of the optimal solution (if available)
    
    """
    
    q,T,coords,omega_max,delta_max = [item for key,item in q_resource.items()]
    
    Q_size = len(Q)

    V = ryd_int(coords)

    encoder = Encoder(q,Q_size,e_layers,omega_max,delta_max,T).to(device)
    decoder = Decoder(q,Q_size,d_layers).to(device)

    #take the upper-triangular elements of the QUBO matrix as the input
    x = Q[torch.triu(torch.ones(Q_size, Q_size) == 1)]

    params = list(encoder.parameters())+list(decoder.parameters())
    optimizer = torch.optim.SGD(params, lr=lr,momentum=0.9)
    
    start = time.time()
  
    solution = []  #store the solution vectors
    loss_list = [] #store the loss values
    for epoch in range(num_epochs):  

        #Ryd denotes the function of quantum simulation
        Ryd = Quantum.apply

        #forward pass
        encoder_out = encoder(x)       
        q_out = Ryd(encoder_out,q,T,V,noise)
        #print(f"q_out:{q_out}")
        y = decoder(q_out)
        loss = qubo_loss(Q,y)

        #backpropagation
        loss.backward()
    
        #update all the parameters 
        optimizer.step()
        
        #clear the gradients 
        optimizer.zero_grad()


        with torch.no_grad():
            sol_vector = decoder(Ryd(encoder(x),q,T,V,noise))
            loss_value = qubo_loss(Q,sol_vector)
            solution.append(sol_vector)
            loss_list.append(loss_value)

            if (epoch+1)%10 == 0:
                if Q_sol == None:
                    print(f"epoch {epoch+1}: Loss = {loss_value.item()}")
                else:
                    print(f"epoch {epoch+1}: Loss = {100*(loss_value.item()-Q_sol)/abs(Q_sol)}%")
                    
    end = time.time()
  
    #map the continuous (probabilistic) output to binary
    binary_solution = [np.array([0 if v <0.5 else 1 for v in solution[i]]) for i in range(num_epochs)]

    binary_losses=[]    #losses evaluated with the binary-variable output
    for bs in binary_solution:
        binary_tensor = torch.tensor(bs,dtype=torch.float32)
        binary_tensor = torch.reshape(binary_tensor,(-1,))
        binary_losses.append(qubo_loss(Q,binary_tensor).item())

    prob_losses = [s.item() for s in loss_list]  #losses evaluated with the continuous-variable output

    # If the optimal solution is not provided, then simply return the loss values
    if Q_sol == None:
        result = [binary_losses,prob_losses]
    # If the optimal solution is provided, then return the percentage error relative to the optimal solution
    else:
        bin_err = [100*abs((s-Q_sol)/Q_sol) for s in binary_losses]
        prob_err = [100*abs((s-Q_sol)/Q_sol) for s in prob_losses]
        result = [bin_err,prob_err]

    print(f"Solution: {binary_solution[-1]}")
    print(f"QUBO cost: {binary_losses[-1]}")
    if Q_sol != None:
        print(f"Percentage Error: {result[0][-1]}%")
    print(f"Solving time: {end-start}s")

    return result 

# Plot the learning curve
def plot_learning_curve(result,figsize=(6,4),logx=False,logy=False):

    num_epochs  = len(result[0])
    plt.figure(figsize=figsize)
    plt.plot(range(1,num_epochs+1),result[1],lw=3,label="continuous variables")
    plt.plot(range(1,num_epochs+1),result[0],lw=3,label="binary variables",ls="dashed",color="tomato")
    plt.ylabel("Loss",fontsize=15)
    plt.xlabel("Iterations",fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    if logx == True:
        plt.xscale("log")
    if logy == True:
        if result[0][-1] < 0:
            raise ValueError("Negative values found in the result. Try percentage error loss instead or disable logy.")
        else:
            plt.yscale("log")
    plt.show()
