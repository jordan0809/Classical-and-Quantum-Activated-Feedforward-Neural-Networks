import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy


class ClassicSolver(nn.Module):
    def __init__(self,Q_size,layers):
        super(ClassicSolver,self).__init__()
        self.layers = layers
        self.hidden = nn.ModuleList()
        for i in range(layers):
            if i ==0:
                self.hidden.append(nn.Linear(4,4+Q_size))
            if 0<i<layers-1:
                self.hidden.append(nn.Linear(4+Q_size,4+Q_size))
            if i == layers-1:
                self.hidden.append(nn.Linear(4+Q_size,Q_size))

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
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
    

def qubo_loss(Q,x):
    off = Q.detach().clone()
    off.fill_diagonal_(0)
    return torch.dot(x,torch.diag(Q))+x.t()@off@x  #linear terms + quadratic terms




def fnn_solve(Q_list,layers,Q_sol_list=None,learning_rate=0.01,num_epochs=500,pre_rounds=20,anneal_round=10,post_annealing=True):

    """
    Q_list: a list of QUBO matrices (type: torch.tensor)
    Q_sol_list: a list of solution cost values
    
    """

    cost_list = []  # a list of minimum cost values
    vector_list = [] # a list of solution vectors
    for index,Q in enumerate(Q_list):

        if Q_sol_list !=None:
            Q_sol = Q_sol_list[index]
    
        Q_size = len(Q)

        #pre-sampling
        fnn_state_list = []
        inputs_list=[]
        pre_loss = []
        for pre_round in range(pre_rounds):  #pre-sampling rounds

            fnn = ClassicSolver(Q_size,layers)
            fnn_state_list.append(fnn.state_dict().copy())

            inputs = torch.rand(4,requires_grad=True)
            inputs_list.append(deepcopy(inputs))
            
            optimizer = torch.optim.SGD(fnn.parameters(), lr=learning_rate)

            for epoch in range(20):  #20 epochs of training for each pre-sampling round

                out = fnn(inputs)
        
                loss = qubo_loss(Q,out)
    
                loss.backward()

                with torch.no_grad():
                    inputs -= learning_rate*inputs.grad
    
                optimizer.step()
           
                #clear the gradients 
                optimizer.zero_grad()
                inputs.grad.data.zero_()

            pre_loss.append(loss.item())

        best_fnn_params = fnn_state_list[np.argmin(pre_loss)]
        best_inputs = inputs_list[np.argmin(pre_loss)]

        #actual training
        fnn = ClassicSolver(Q_size,layers)
        fnn.load_state_dict(best_fnn_params)

        inputs = best_inputs
        inputs.requires_grad=True
    
        optimizer = torch.optim.SGD(fnn.parameters(), lr=learning_rate)
    
        solution = []
        for epoch in range(num_epochs):

            out = fnn(inputs)
        
            loss = qubo_loss(Q,out)
    
            loss.backward()

            with torch.no_grad():
                inputs -= learning_rate*inputs.grad
    
            optimizer.step()
           
            #clear the gradients 
            optimizer.zero_grad()
            inputs.grad.data.zero_()
            
            with torch.no_grad():
                out = fnn(inputs)
                solution.append(out)

        
        binary_solution = [np.array([0 if v <0.5 else 1 for v in solution[i]]) for i in range(num_epochs)]
        fnn_sol = binary_solution[-1]

        W = Q.detach().numpy()
        
        best = fnn_sol.T@W@fnn_sol   #cost value
        best_sol = fnn_sol   #solution vector
        
        print("fnn loss:",best)
        if post_annealing:
            u = np.identity(Q_size)

            T0 = abs(best)/Q_size
            for a in range(1,anneal_round+1):
                print(f"annealing round: {a}/{anneal_round}")
                T = T0/a
                for i in range(Q_size):
                    trial_sol = abs(best_sol - u[i])
                    loss = trial_sol.T@W@trial_sol
                    if loss < best:
                        
                        best_sol = trial_sol
                        best = loss
                        print("better solution found, loss:",loss)
                    else:
                        if np.random.rand() < np.exp((best-loss)/T):
                            
                            for j in range(Q_size):
                                if j != i:
                                    trial_sol2 = abs(trial_sol-u[j])
                                    loss = trial_sol2.T@W@trial_sol2
                                    if loss < best:
                                        best_sol =trial_sol2
                                        best = loss
                                        print("energy barrier overcome, loss:",loss)

        if Q_sol_list !=None:
            percentage_error = 100*(best-Q_sol)/abs(Q_sol)
            if percentage_error > 0.001:
                cost_list.append(percentage_error)
            else:
                cost_list.append(0)
            print(f"instance: {index+1}/{len(Q_list)}, loss={percentage_error}")
        else:
            cost_list.append(best)
            print(f"instance: {index+1}/{len(Q_list)}, loss={best}")

        vector_list.append(best_sol)
        
    return cost_list,vector_list
