# Classical-and-Quantum-Activated-Feedforward-Neural-Networks
This repository contains the code and datasets used in the paper [Towards arbitrary QUBO optimization: analysis of classical and quantum-activated feedforward neural networks] (https://iopscience.iop.org/article/10.1088/2632-2153/addb97
).
It includes QUBO datasets, implementations of FNN and QCED models, and interfaces for classical and quantum optimizers including Gurobi, D-Wave Advantage, and the Fujitsu Digital Annealer emulator.

To install the necessary packages for implementing the FNN and QCED optimizers, please follow the `pyproject.toml` file and run:

```bash
pip install . 
```

For the implementation of the Gurobi solver:

1. Install Gurobi from [https://www.gurobi.com](https://www.gurobi.com).  
2. Activate your license (free academic licenses are available).  
3. Install the optional dependency:

```bash
pip install ".[gurobi]"
```

For the implementation of the D-Wave quantum annealer: 

1. Create an account on the D-Wave Leap platform: [https://cloud.dwavesys.com/leap](https://cloud.dwavesys.com/leap)
2. Create a project and obtain your API token from the Leap dashboard.
3. Install the optional dependency:

```bash
pip install ".[dwave]"
```

For the implementation of the Fujitsu Digital Annealer emulator, please follow the tutorial in `Fujitsu_Digital_Annealer_Tutorial.zip`.
