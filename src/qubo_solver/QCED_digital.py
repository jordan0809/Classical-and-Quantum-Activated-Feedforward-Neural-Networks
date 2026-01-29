import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def qaoa_ansatz(num_qubits: int, num_layers: int):
    """Representative quantum ansatz: QAOA"""
    circ = QuantumCircuit(num_qubits)

    m_params = ParameterVector("mixer", num_layers)
    c_params = ParameterVector(
        "cost", num_layers * (num_qubits * (num_qubits - 1) // 2)
    )

    circ.h(range(num_qubits))

    cost_idx = 0
    for layer in range(num_layers):
        circ.rx(m_params[layer], range(num_qubits))

        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                circ.rzz(c_params[cost_idx], i, j)
                cost_idx += 1

    params = list(circ.parameters)
    return circ, params


def quantum_activation(ansatz: QuantumCircuit, assigned_vals: list[float]):
    """Quantum activation function implemented as a quantum circuit. The output values are expectation values of the parities for each pair of qubits."""

    n = ansatz.num_qubits

    # evaluate pairwise parity
    paulis = [[("ZZ", [i, j], 1)] for i in range(n) for j in range(i + 1, n)]
    observables = [SparsePauliOp.from_sparse_list(p, n) for p in paulis]

    estimator = EstimatorV2()
    pubs = [(ansatz, observables, assigned_vals)]
    job = estimator.run(pubs)
    result = job.result()[0]
    expectations = torch.tensor(result.data.evs.tolist(), dtype=torch.float32)

    return expectations


def parameter_shift(ansatz: QuantumCircuit, assigned_vals: list[float]):
    """Using the parameter-shift rule for gradient calculation."""
    n = ansatz.num_qubits
    num_params = ansatz.num_parameters
    num_observe = n * (n - 1) // 2
    gradients = torch.zeros(num_observe, num_params, dtype=torch.float32)

    # prepare shifted parameter vectors (+ and -) for all parameters
    all_shifted_vals = np.zeros((2 * num_params, num_params))

    shift = np.pi / 2
    for i in range(num_params):
        plus_vals = assigned_vals.copy()
        plus_vals[i] += shift
        minus_vals = assigned_vals.copy()
        minus_vals[i] -= shift

        all_shifted_vals[2 * i] = plus_vals
        all_shifted_vals[2 * i + 1] = minus_vals

    # compute expectation values for all parameter vectors in parallel
    paulis = [[("ZZ", [i, j], 1)] for i in range(n) for j in range(i + 1, n)]
    observables = [SparsePauliOp.from_sparse_list(p, n) for p in paulis]

    estimator = EstimatorV2()
    # reshape for batch processing (# of parameter vectors, 1 (broadcast over all observables), # of parameters)
    # evaluate all observables for all sets of parameter vectors.
    batch_shifted_vals = all_shifted_vals.reshape(2 * num_params, 1, num_params)
    pubs = [(ansatz, observables, batch_shifted_vals)]
    job = estimator.run(pubs)
    result = job.result()[0]
    expectations = result.data.evs

    for i in range(num_params):
        diff = expectations[2 * i] - expectations[2 * i + 1]
        gradients[:, i] = torch.tensor(diff / 2, dtype=torch.float32)

    return gradients


class QuantumActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, ansatz):
        # save for backward pass
        ctx.ansatz = ansatz
        ctx.save_for_backward(input_tensor)

        # convert tensor to numpy for Qiskit
        values = input_tensor.detach().numpy()

        expectations = quantum_activation(ansatz, values)

        return expectations

    @staticmethod
    def backward(ctx, grad_output):
        (input_tensor,) = ctx.saved_tensors
        ansatz = ctx.ansatz
        values = input_tensor.detach().numpy().flatten()

        grad_output_input = parameter_shift(
            ansatz, values
        )  # shape [num_outputs, num_inputs]
        grad_loss_output = torch.reshape(
            grad_output, (1, grad_output.shape[0])
        )  # shape [1,num_outputs]
        grad_loss_input = torch.matmul(
            grad_loss_output, grad_output_input
        )  # shape [1, num_inputs]
        gradients = grad_loss_input.flatten()  # flatten to match the shape of input

        return gradients, None


class QuantumLayer(nn.Module):
    """Quantum layer of QCED_digital"""

    def __init__(self, ansatz):
        super().__init__()
        self.ansatz = ansatz

    def forward(self, x):
        return QuantumActivation.apply(x, self.ansatz)


class Encoder(nn.Module):
    """Encoding layer of QCED_digital"""

    def __init__(self, input_size, output_size, hidden_layers):
        super().__init__()
        self.layers = hidden_layers
        self.hidden = nn.ModuleList()
        num_neurons = np.linspace(input_size, output_size, hidden_layers + 1)
        for i in range(hidden_layers):
            self.hidden.append(nn.Linear(int(num_neurons[i]), int(num_neurons[i + 1])))

        self.relu = nn.ReLU()

    def forward(self, x):
        out = x
        for i in range(self.layers):
            func = self.hidden[i]
            out = func(out)
            out = self.relu(out)

        return out


class Decoder(nn.Module):
    """Decoding layer of QCED_digital"""

    def __init__(self, input_size, Q_size, hidden_layers):
        super().__init__()
        self.layers = hidden_layers
        self.hidden = nn.ModuleList()
        num_neurons = np.linspace(input_size, Q_size, hidden_layers + 1)
        for i in range(hidden_layers):
            self.hidden.append(nn.Linear(int(num_neurons[i]), int(num_neurons[i + 1])))

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = x
        for i in range(self.layers):
            func = self.hidden[i]
            out = func(out)
            if i < self.layers - 1:
                out = self.relu(out)
            else:
                out = (1 + self.tanh(out)) / 2

        return out


def qubo_loss(Q, x):
    """Evaluate the QUBO loss function"""
    off = Q.detach().clone()
    off.fill_diagonal_(0)
    return (
        torch.dot(x, torch.diag(Q)) + x.t() @ off @ x
    )  # linear terms + quadratic terms


def QCED_digital_solve(
    Q: Tensor,
    qansatz: QuantumCircuit | None = None,
    num_epochs: int = 100,
    lr: float = 0.01,
    e_layers: int = 3,
    d_layers: int = 3,
    Q_sol: float | None = None,
):
    """
    QCED optimizer with a digital (circuit-based) quantum activation function.
    The quantum ansatz (`qansatz`) can be a customized parameterized circuit.
    If `qansatz` is None, a default QAOA-based parameterized circuit is used.
    """
    if qansatz is None:
        qaoa, _ = qaoa_ansatz(4, 3)
        qansatz = qaoa

    input_vector = Q[torch.triu(torch.ones(len(Q), len(Q)) == 1)]
    input_size = len(input_vector)
    output_size = qansatz.num_parameters

    test_weights = np.random.rand(output_size)
    expect = quantum_activation(qansatz, test_weights)
    decoder_input_size = len(expect)

    encoder = Encoder(input_size, output_size, e_layers).to(device)
    qlayer = QuantumLayer(qansatz)
    decoder = Decoder(decoder_input_size, len(Q), d_layers).to(device)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)

    start = time.time()

    solutions = []  # store the solution vectors
    loss_list = []  # store the loss values
    for epoch in range(num_epochs):
        # forward pass
        qparams = encoder(input_vector)
        qout = qlayer(qparams)
        output = decoder(qout)

        # backward pass
        loss = qubo_loss(Q, output)
        loss.backward()

        # update all the parameters
        optimizer.step()

        # clear the gradients
        optimizer.zero_grad()

        with torch.no_grad():
            sol_vector = decoder(qlayer(encoder(input_vector)))
            loss_value = qubo_loss(Q, sol_vector)
            solutions.append(sol_vector)
            loss_list.append(loss_value)

            if (epoch + 1) % 10 == 0:
                if Q_sol is None:
                    print(f"epoch {epoch + 1}: Loss = {loss_value.item()}")
                else:
                    print(
                        f"epoch {epoch + 1}: Loss = {100 * (loss_value.item() - Q_sol) / abs(Q_sol)}%"
                    )

    end = time.time()

    binary_solution = [
        np.array([0 if v < 0.5 else 1 for v in solutions[i]]) for i in range(num_epochs)
    ]

    binary_losses = []  # losses evaluated with the binary output
    for bs in binary_solution:
        binary_tensor = torch.tensor(bs, dtype=torch.float32)
        binary_tensor = torch.reshape(binary_tensor, (-1,))
        binary_losses.append(qubo_loss(Q, binary_tensor).item())

    prob_losses = [
        s.item() for s in loss_list
    ]  # losses evaluated with the continuous output

    # If the optimal solution is not provided, the loss values are returned
    if Q_sol is None:
        result = [binary_losses, prob_losses]
    # If the optimal solution is provided, the percentage errors relative to the optimal solution are returned
    else:
        bin_err = [100 * abs((s - Q_sol) / Q_sol) for s in binary_losses]
        prob_err = [100 * abs((s - Q_sol) / Q_sol) for s in prob_losses]
        result = [bin_err, prob_err]

    print(f"Solution: {binary_solution[-1]}")
    print(f"QUBO cost: {binary_losses[-1]}")
    if Q_sol is not None:
        print(f"Percentage Error: {result[0][-1]}%")
    print(f"Solving time: {end - start}s")

    return result
