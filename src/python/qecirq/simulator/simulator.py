import numpy as np
from itertools import cycle
from openfermion.ops import QubitOperator, IsingOperator
from openfermion import get_sparse_operator

from zquantum.core.openfermion import expectation, change_operator_type
from zquantum.core.interfaces.backend import QuantumSimulator
from zquantum.core.measurement import (
    expectation_values_to_real,
    ExpectationValues,
    Measurements,
)
from zquantum.core.estimator import get_context_selection_circuit

from cirq import Circuit, measure, Simulator, measure_each
from cirq import DensityMatrixSimulator
from cirq import generalized_amplitude_damp




class CirqSimulator(QuantumSimulator):
    def __init__(
        self,
        n_samples=None,
        noise_model=None,
        **kwargs,
    ):
        """ Get a cirq device (simulator or QPU) that adheres to the 
        zquantum.core.interfaces.backend.QuantumSimulator
        Args:
            simulator (string):
            device_name (string): the name of the device
            n_samples (int): the number of samples to use when running the device
            noise_model (qiskit.providers.aer.noise.NoiseModel): an optional #TODO: should be a cirq noise model
                noise model to pass in for noisy simulations
        Returns:
            qecirq.backend.CirqSimulator
        """
        self.n_samples = n_samples
        self.noise_model = noise_model
        self.num_circuits_run = 0
        self.num_jobs_run = 0
        if self.noise_model is not None:
            self.simulator = DensityMatrixSimulator(dtype = np.complex128)
        else:
            self.simulator = Simulator()

    def run_circuit_and_measure(self, circuit, **kwargs):
        """ Run a circuit and measure a certain number of bitstrings. Note: the
        number of bitstrings measured is derived from self.n_samples
        Args:
            circuit (zquantum.core.circuit.Circuit): the circuit to prepare the state
        Returns:
            a list of bitstrings (a list of tuples)
        """
        self.num_circuits_run += 1
        self.num_jobs_run += 1
        num_qubits = len(circuit.qubits)
        cirq_circuit = circuit.to_cirq()
        if self.noise_model is not None:
            cirq_circuit.with_noise(self.noise_model)

        qubits = list(cirq_circuit.all_qubits())
        for i in range(0, len(qubits)):
            cirq_circuit.append(measure_each(qubits[i]))

        result_object = self.simulator.run(cirq_circuit, repetitions = self.n_samples)
        measurement = get_measurement_from_cirq_result_object(result_object, qubits)

        return measurement

    def run_circuitset_and_measure(self, circuitset, **kwargs):
        """ Run a set of circuits and measure a certain number of bitstrings.
        Note: the number of bitstrings measured is derived from self.n_samples
        Args:
            circuit (zquantum.core.circuit.Circuit): the circuit to prepare the state
        Returns:
            a list of lists of bitstrings (a list of lists of tuples)
        """
        cirq_circuitset = []
        measurements_set = []
        qubit_listset = []
        for circuit in circuitset:
            num_qubits = len(circuit.qubits)
            cirq_circuit = circuit.to_cirq()
            if self.noise_model is not None:
                cirq_circuit.with_noise(self.noise_model)
            qubits = list(cirq_circuit.all_qubits())
            for i in range(0, len(qubits)):
                cirq_circuit.append(measure_each(qubits[i]))
            cirq_circuitset.append(cirq_circuit)
            qubit_listset.append(qubits)
        result = self.simulator.run_batch(cirq_circuitset, repetitions= self.n_samples)

        for i in range(len(cirq_circuitset)):
            measurements = get_measurement_from_cirq_result_object(result[i][0], qubit_listset[i])
            measurements_set.append(measurements)   
 
        return measurements_set
 
    def get_exact_expectation_values(self, circuit, qubit_operator, **kwargs):
        """ Run a circuit to prepare a wavefunction and measure the exact 
        expectation values with respect to a given operator.
        Args:
            circuit (zquantum.core.circuit.Circuit): the circuit to prepare the state
            qubit_operator (openfermion.ops.QubitOperator): the operator to measure
        Returns:
            zquantum.core.measurement.ExpectationValues: the expectation values
                of each term in the operator
        """
        self.num_circuits_run += 1
        self.num_jobs_run += 1

        if self.noise_model is not None:
            return self.get_exact_noisy_expectation_values(circuit, qubit_operator, **kwargs)
        else:
            wavefunction = self.get_wavefunction(circuit)
            n_qubits = len(circuit.qubits)

            # Pyquil does not support PauliSums with no terms.
            if len(qubit_operator.terms) == 0:
                return ExpectationValues(np.zeros((0,)))

            values = []

            for pauli_term in qubit_operator:
                sparse_pauli_term_ndarray = get_sparse_operator(pauli_term, n_qubits=n_qubits).toarray()
                if np.size(sparse_pauli_term_ndarray) == 1:
                    expectation_value = sparse_pauli_term_ndarray[0][0]
                    values.append(expectation_value)
                else:
                    expectation_value = np.real(wavefunction.conj().T @ sparse_pauli_term_ndarray @ wavefunction)
                    values.append(expectation_value)

            return expectation_values_to_real(ExpectationValues(np.asarray(values)))

    def get_exact_noisy_expectation_values(self, circuit, qubit_operator, **kwargs):
        """ Run a circuit to prepare a wavefunction and measure the exact 
        expectation values with respect to a given operator.
        Args:
            circuit (zquantum.core.circuit.Circuit): the circuit to prepare the state
            qubit_operator (openfermion.ops.QubitOperator): the operator to measure
        Returns:
            zquantum.core.measurement.ExpectationValues: the expectation values
                of each term in the operator
        """
        if self.noise_model is None:
            raise RuntimeError("Please provide noise model to get exact noisy expectation values")
        else:
            cirq_circuit = circuit.to_cirq()
            values = []
            n_qubits = len(circuit.qubits)
            for pauli_term in qubit_operator:
                sparse_pauli_term_ndarray = get_sparse_operator(pauli_term, n_qubits=n_qubits).toarray()     
                if np.size(sparse_pauli_term_ndarray) == 1:
                    expectation_value = sparse_pauli_term_ndarray[0][0]
                    values.append(expectation_value)
                else:
                    noisy_circuit = cirq_circuit.with_noise(self.noise_model)
                    rho = self.simulator.simulate(noisy_circuit).final_density_matrix
                    expectation_value = np.real(np.trace(rho @ sparse_pauli_term_ndarray))
                    values.append(expectation_value)
        return expectation_values_to_real(ExpectationValues(np.asarray(values)))

    def get_wavefunction(self, circuit):
        """ Run a circuit and get the wavefunction of the resulting statevector.
        Args:
            circuit (zquantum.core.circuit.Circuit): the circuit to prepare the state
        Returns:
            wavefunction (ndarray): The wavefunction representing the circuit
        """

        wavefunction = circuit.to_cirq().final_state_vector()
 
        return wavefunction

def get_measurement_from_cirq_result_object(result_object, qubits):
    """Gets measurement bit strings from cirq result object and returns a Measurement object

    Args:


    Return:
        measurment (zquantum.core.measurement.Measurements)

    """

    keys = list(range(len(qubits)))

    numpy_samples = list(zip(*(result_object._measurements[str(sub_key)]
                    for sub_key in keys)))
    
    samples = []
    for numpy_bitstring in numpy_samples:
        bitstrings = []
        for key in numpy_bitstring:
            bitstrings.append(key[0])
        samples.append(tuple(bitstrings))
    
    measurement = Measurements() 
    measurement.bitstrings = samples

    return measurement
