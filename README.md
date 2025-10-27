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

For the implementation of the Fujitsu Digital Annealer emulator:

1. Install Python 3.9 and Jupyter 
2. Download `dadk_light_3.9.tar.bz2` from Fujitsu
3. Install manually:

```bash
pip install -U Software/dadk_light_3.9.tar.bz2
```

or follow the `fujitsu_environment.yml` file