# Classical-and-Quantum-Activated-Feedforward-Neural-Networks
QUBO datasets, FNN, QCED, Gurobi, D-Wave Advantage, and Fujitsu Digital Annealer emulator. 
To install the necessary packages for implementing all solvers except the Fujitsu Digital Annealer emulator, please follow the `pyproject.toml` file.

For the implementation of the Gurobi solver:

1. Install Gurobi from [https://www.gurobi.com](https://www.gurobi.com).  
2. Activate your license (free academic licenses are available).  
3. Inside the activated conda environment, run:

```bash
pip install gurobipy
```

For the implementation of the Fujitsu Digital Annealer emulator, please follow the tutorial in `Fujitsu_Digital_Annealer_Tutorial.zip`.
