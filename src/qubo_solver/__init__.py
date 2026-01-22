try:
    from qubo_solver.Dwave import dwave_solve
except ImportError:
    dwave_solve = None
try:
    from qubo_solver.Fujitsu import fujitsu_solve
except ImportError:
    fujitsu_solve = None
try:
    from qubo_solver.Gurobi import gurobi_solve
except ImportError:
    gurobi_solve = None
try:
    from qubo_solver.QCED_digital import QCED_digital_solve
except ImportError:
    QCED_digital_solve = None

from qubo_solver.FNN import fnn_solve
from qubo_solver.QCED import QCED_solve, plot_learning_curve

__all__ = [
    "dwave_solve",
    "fnn_solve",
    "fujitsu_solve",
    "gurobi_solve",
    "QCED_digital_solve",
    "QCED_solve",
    "plot_learning_curve",
]
