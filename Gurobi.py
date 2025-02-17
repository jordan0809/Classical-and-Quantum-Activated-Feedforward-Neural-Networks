import gurobipy as gp
from gurobipy import GRB
import numpy as np

def gurobi_solve(Q):
    n = len(Q)
    
    # Create a new model
    model = gp.Model("QUBO")

    # Create variables
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    
    # Create auxiliary variables and add constraints to linearize quadratic terms
    y = {}
    for i in range(n):
        for j in range(i + 1, n):
            y[i, j] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")
            model.addConstr(y[i, j] <= x[i])
            model.addConstr(y[i, j] <= x[j])
            model.addConstr(y[i, j] >= x[i] + x[j] - 1)
    
    # Set the objective
    obj = gp.quicksum(2*Q[i, j] * y[i, j] for i in range(n) for j in range(i+1,n))
    obj += gp.quicksum(Q[i, i] * x[i] for i in range(n))
    model.setObjective(obj, GRB.MINIMIZE)

    model.Params.TimeLimit = 100  #limit the time to 100 seconds per instance (free to adjust)

    # Optimize the model
    model.optimize()

    # Retrieve the solution
    solution = [x[i].x for i in range(n)]
    best_value = model.objVal
    #return solution
    return best_value,solution
