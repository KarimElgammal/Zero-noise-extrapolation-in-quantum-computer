from qiskit import QuantumCircuit, transpile
from qiskit_aer.noise import NoiseModel, depolarizing_error
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def depolarizing_noise_model(p_error_1q, p_error_2q):
    '''
    function that creates a noise model with depolarizing error
    '''
    noise_model = NoiseModel()
    # apply on 1qubit gates
    error_1q = depolarizing_error(p_error_1q, 1)
    noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'rx', 'rz'])
    # apply on 2qubit gates
    error_2q = depolarizing_error(p_error_2q, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
    return noise_model

def generate_GHZ(n_qubits):
    """
    Generate a GHZ state with n qubits
    """
    circ = QuantumCircuit(n_qubits)
    circ.h(0)
    for qubit in range(n_qubits - 1):
        circ.cx(qubit, qubit + 1)
    circ.measure_all()
    return circ

def generate_layered_circuit(num_layers, num_qubits, rotation_angles):
    """
    Generate a layered quantum circuit with num_layers and num_qubits
    it is a general circuit with Rz and CNOT gates
    """
    qc = QuantumCircuit(num_qubits)
    for l in range(num_layers):
        for i in range(num_qubits):
            qc.rz(rotation_angles[l, i], i)
        for i in range(num_qubits-1):
            qc.cx(i, i+1)
    return qc


def probabilities(counts, dim):
    """
    Function that calculates the probabilities of the states by converting the counts to probabilities
    """
    prob = [counts.get(j, 0) for j in range(dim)]
    prob = np.array(prob)
    prob = prob/np.sum(prob)
    return prob


def unitary_folding(num_folds, num_layers, num_qubits, rotation_angles, simulator, state_vector_dim, print_circuit=False):
    """
    Applies unitary folding to a quantum circuit and returns the noisy probabilities
    """
    #print(f'Folding {num_folds} times')
    circuit = generate_layered_circuit(num_layers, num_qubits, rotation_angles)
    folded_circuit = fold_circuit(circuit, num_folds, simulator)
    if print_circuit:
        print(f'Folded circuit: {folded_circuit}')
    counts = run_circuit(folded_circuit, simulator)
    counts_int = {int(state, 2): count for state, count in counts.items()}
    prob_noise = probabilities(counts_int, state_vector_dim)
    return prob_noise

def fold_circuit(circuit, num_folds, simulator):
    """
    Folds a quantum circuit by composing it with itself num_folds times
    """
    transpiled_circuit = transpile(circuit, simulator)
    for _ in range(num_folds - 1):
        transpiled_circuit = transpiled_circuit.compose(transpiled_circuit)
    return transpiled_circuit

def run_circuit(circuit, simulator):
    """
    Runs a quantum circuit on a simulator and returns the counts
    """
    circuit.measure_all()
    result = simulator.run(circuit).result()
    return result.get_counts()

def fit_polynomial(x, y, degree):
    """
    Fits a polynomial of the given degree to the data and returns the polynomial function
    """
    coefficients = np.polyfit(x, y, degree)
    return np.poly1d(coefficients)

def fit_exponential(x, y):
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c
    popt, pcov = curve_fit(func, x, y, maxfev=5000)
    return lambda k: func(k, *popt)

def fit_linear(x, y):
    p = np.polyfit(x, y, 1)
    return np.poly1d(p)

def plot_probabilities(params):
    """
    Plots the probabilities of the outcomes of a quantum circuit
    """
    try:
        plt.figure(figsize=(4, 4))
        # Plot the ideal probability point
        plt.plot(0, params['ideal_probability'], marker="o", markersize=10, color="green", label='ideal')
        
        # Fit a function to the probabilities and plot it
        fitted_function = params['fit_function'](np.arange(1, params['num_folds']), params['probabilities'])
        plt.plot([fitted_function(k) for k in np.arange(params['num_folds'])], color="blue")

        # Plot the probabilities        
        plt.scatter(np.arange(1, params['num_folds']), params['probabilities'], color="blue", label="error mitigation")
        
        plt.title(f'Outcome "{params["outcome"]}" probability after error mitigation ({params["fit_type"]} fit): {round(fitted_function(0),2)}')
        plt.ylabel(f'Probability for outcome "{params["outcome"]}"')
        plt.xlabel(f'Number of circuit repetitions for "{params["outcome"]}"')
        
        print(f"Ideal probability for outcome '{params['outcome']}': {params['ideal_probability']}")
        print(f"Probability without error mitigation for outcome '{params['outcome']}': {params['unmitigated_probability']}")
        print(f"Probability with error mitigation for outcome '{params['outcome']}': {round(fitted_function(0), 2)}")
        
        plt.grid(True)
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"An error occurred for outcome {params['outcome']}: {e}")