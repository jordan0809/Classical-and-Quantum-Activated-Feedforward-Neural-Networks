# Classical-and-Quantum-Activated-Feedforward-Neural-Networks
This repository contains the code and datasets used in the paper **Towards arbitrary QUBO optimization: analysis of classical and quantum-activated feedforward neural networks** (https://iopscience.iop.org/article/10.1088/2632-2153/addb97
).
It includes QUBO datasets, implementations of FNN and QCED models, and interfaces for classical and quantum optimizers including Gurobi, D-Wave Advantage, and the Fujitsu Digital Annealer emulator.


## Installation

This project requires **Python 3.9 or 3.10**.

To install the necessary packages for implementing the FNN and QCED optimizers (`FNN.py` and `QCED.py`), please follow the `pyproject.toml` file and run:

```bash
pip install -e . 
```

To experiment with digital (circuit-based) quantum activation functions (`QCED_digital.py`), which uses *Qiskit* and *EstimatorV2* for circuit simulation:

```bash
pip install -e ".[qiskit]"
```

**Note**: The QCED optimizer proposed in the paper leverages a quantum activation layer based on the simulation of an analog system (i.e. Rydberg annealer) using *QuTiP*. The digital variant uses *Qiskit* as the simulation backend for gate-based execution.

For the implementation of the Gurobi solver (`Gurobi.py`):

1. Install Gurobi from [https://www.gurobi.com](https://www.gurobi.com).  
2. Activate your license (free academic licenses are available).  
3. Install the optional dependency:

```bash
pip install ".[gurobi]"
```

For the implementation of the D-Wave quantum annealer (`Dwave.py`): 

1. Create an account on the D-Wave Leap platform: [https://cloud.dwavesys.com/leap](https://cloud.dwavesys.com/leap)
2. Create a project and obtain your API token from the Leap dashboard.
3. Install the optional dependency:

```bash
pip install ".[dwave]"
```

For the implementation of the Fujitsu Digital Annealer emulator (`Fujitsu.py`), please follow the tutorial in `Fujitsu_Digital_Annealer_Tutorial.zip`.

## Usage
For a quick start and example usage, please refer to the provided Jupyter Notebook:

[`example/example.ipynb`](example/example.ipynb)