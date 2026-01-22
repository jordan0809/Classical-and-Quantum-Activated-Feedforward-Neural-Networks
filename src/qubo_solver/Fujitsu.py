from dadk.BinPol import BinPol, Term, QUBOSolverCPU, GraphicsDetail, AutoTuning
import numpy as np


def fujitsu_solve(Q):
    Q = Q.detach().numpy()
    qubo = BinPol.sum(
        Term(Q[i, j], (i, j)) for i in range(len(Q)) for j in range(len(Q))
    )
    solver = QUBOSolverCPU(
        number_iterations=qubo.N**2,  # Total number of iterations per run.
        number_runs=4,  # Number of stochastically independent runs.
        temperature_start=0.01,  # Start temperature of the annealing process.
        temperature_end=0.0001,  # End temperature of the annealing process.
        temperature_mode=0,  # 0, 1, or 2 to define the cooling curve:
        #    0, 'EXPONENTIAL':
        #       reduce temperature by factor (1-temperature_decay) every temperature_interval steps
        #    1, 'INVERSE':
        #       reduce temperature by factor (1-temperature_decay*temperature) every temperature_interval steps
        #    2, 'INVERSE_ROOT':
        #       reduce temperature by factor (1-temperature_decay*temperature^2) every temperature_interval steps
        temperature_interval=qubo.N,  # Number of iterations keeping temperature constant.
        offset_increase_rate=0.00005,  # Increase of dynamic offset when no bit selected. Set to 0.0 to switch off dynamic energy feature.
        bit_precision=64,  # Bit precision (DAU version 2).
        graphics=GraphicsDetail.ALL,  # Switch on graphics output.
        auto_tuning=AutoTuning.AUTO_SCALING,  # Following methods for scaling ``qubo`` and determining temperatures are available:
        #    AutoTuning.NOTHING:
        #       no action
        #    AutoTuning.SCALING:
        #       ``scaling_factor`` is multiplied to ``qubo``, ``temperature_start``, ``temperature_end`` and ``offset_increase_rate``.
        #    AutoTuning.AUTO_SCALING:
        #       A maximum scaling factor w.r.t. ``scaling_bit_precision`` is multiplied to ``qubo``, ``temperature_start``, ``temperature_end`` and ``offset_increase_rate``.
        #    AutoTuning.SAMPLING:
        #       ``temperature_start``, ``temperature_end`` and ``offset_increase_rate`` are automatically determined.
        #    AutoTuning.AUTO_SCALING_AND_SAMPLING:
        #       Temperatures and scaling factor are automatically determined and applied.
        scaling_bit_precision=62,  # Maximum bit_precision for ``qubo``. Used to define the scaling factor for ``qubo``, ``temperature_start``, ``temperature_end`` and ``offset_increase_rate``.
    )
    solution_list = solver.minimize(qubo)
    solution = np.array(solution_list.solutions[0].configuration)
    cost = solution.T @ Q @ solution

    return cost, solution
