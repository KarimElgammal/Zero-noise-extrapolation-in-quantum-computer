# Zero-Noise Extrapolation (ZNE) implementation for error mitigation in quantum simulations

This repository contains an implementation of Zero-Noise Extrapolation (ZNE) for error mitigation in quantum circuits using `Qiskit`. It can be good example for learning purposes, but you should consider a dedicated package like [mitiq](https://mitiq.readthedocs.io/en/stable/guide/zne-5-theory.html) for that. The approach here uses unitary folding to simulate the noise and then extrapolates the results to estimate the outcomes.


## The repository files are:

1. [custom ZNE `qiskit` simulation implementation](custom_zne_qiskit.ipynb) contains workflow of using unitary folding to simulate the noise and then extrapolates the results to estimate the outcomes

1. [custom ZNE `qiskit` on IBMQ ](custom_zne_IBMQ_not_working.ipynb) not successful after, maybe need different way to `transile` to accomodate the new changes

- [`aux_functions.py`](aux_functions.py): This file contains auxiliary functions for generating quantum circuits, creating a noise model, running circuits on a simulator, and plotting the results.

## Functions

- `depolarizing_noise_model(p_error_1q, p_error_2q)`: Creates a noise model with depolarizing error.
- `generate_GHZ(n_qubits)`: Generates a GHZ state with n qubits.
- `generate_layered_circuit(num_layers, num_qubits, rotation_angles)`: Generates a layered quantum circuit with num_layers and num_qubits.
- `probabilities(counts, dim)`: Calculates the probabilities of the states by converting the counts to probabilities.
- `unitary_folding(num_folds, num_layers, num_qubits, rotation_angles, simulator, state_vector_dim, print_circuit=False)`: Applies unitary folding to a quantum circuit and returns the noisy probabilities.
- `fold_circuit(circuit, num_folds, simulator)`: Folds a quantum circuit by composing it with itself num_folds times.
- `run_circuit(circuit, simulator)`: Runs a quantum circuit on a simulator and returns the counts.
- `fit_polynomial(x, y, degree)`: Fits a polynomial of the given degree to the data and returns the polynomial function.
- `fit_exponential(x, y)`: Fits an exponential function to the data and returns the function.
- `fit_linear(x, y)`: Fits a linear function to the data and returns the function.
- `plot_probabilities(params)`: Plots the probabilities of the outcomes of a quantum circuit.

## Usage

To use these functions, import them into your Python script:

```python
from aux_functions import generate_GHZ
ghz_circuit = generate_GHZ(3) #generating 3qubit GHZ circuit
```
### packages needed
 `qiskit`, `qiskit-aer`, `numpy`, `scipy`

### Installation
To install the required libraries, run the following command:
```bash
pip install qiskit qiskit-aer numpy matplotlib scipy ipykernel
```

Reference:
- [Building noise models examples using qiskit](https://docs.quantum.ibm.com/verify/building_noise_models)