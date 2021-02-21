from typing import Iterable
import numpy as np
from itertools import cycle
from openfermion.ops import QubitOperator, IsingOperator
from openfermion import get_sparse_operator

from zquantum.core.openfermion import change_operator_type
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
from pyquil.wavefunction import Wavefunction
import sys


class CirqSimulator(QuantumSimulator):
    supports_batching = True
    batch_size = sys.maxsize

    def __init__(
        self,
        n_samples=None,
        noise_model=None,
        **kwargs,
    ):
        """Get a cirq device (simulator or QPU) that adheres to the
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
        super().__init__(n_samples)
        self.noise_model = noise_model
        if self.noise_model is not None:
            self.simulator = DensityMatrixSimulator(dtype=np.complex128)
        else:
            self.simulator = Simulator()

    def run_circuit_and_measure(self, circuit, n_samples=None, **kwargs):
        """Run a circuit and measure a certain number of bitstrings. Note: the
        number of bitstrings measured is derived from self.n_samples
        Args:
            circuit (zquantum.core.circuit.Circuit): the circuit to prepare the state
        Returns:
            a list of bitstrings (a list of tuples)
        """
        super().run_circuit_and_measure(circuit)
        if n_samples is None:
            n_samples = self.n_samples
        cirq_circuit = circuit.to_cirq()
        if self.noise_model is not None:
            cirq_circuit.with_noise(self.noise_model)

        qubits = list(cirq_circuit.all_qubits())
        for i in range(0, len(qubits)):
            cirq_circuit.append(measure_each(qubits[i]))

        result_object = self.simulator.run(cirq_circuit, repetitions=n_samples)
        measurement = get_measurement_from_cirq_result_object(result_object, qubits)

        return measurement

    def run_circuitset_and_measure(self, circuitset, n_samples=None, **kwargs):
        """Run a set of circuits and measure a certain number of bitstrings.
        Note: the number of bitstrings measured is derived from self.n_samples
        Args:
            circuit (zquantum.core.circuit.Circuit): the circuit to prepare the state
        Returns:
            a list of lists of bitstrings (a list of lists of tuples)
        """
        super().run_circuitset_and_measure(circuitset)

        if n_samples is None:
            n_samples = [self.n_samples for circuit in circuitset]
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
        result = self.simulator.run_batch(cirq_circuitset, repetitions=n_samples)

        for i in range(len(cirq_circuitset)):
            measurements = get_measurement_from_cirq_result_object(
                result[i][0], qubit_listset[i]
            )
            measurements_set.append(measurements)

        return measurements_set

    def get_exact_expectation_values(self, circuit, qubit_operator, **kwargs):
        """Run a circuit to prepare a wavefunction and measure the exact
        expectation values with respect to a given operator.
        Args:
            circuit (zquantum.core.circuit.Circuit): the circuit to prepare the state
            qubit_operator (openfermion.ops.QubitOperator): the operator to measure
        Returns:
            zquantum.core.measurement.ExpectationValues: the expectation values
                of each term in the operator
        """
        if self.noise_model is not None:
            return self.get_exact_noisy_expectation_values(
                circuit, qubit_operator, **kwargs
            )
        else:
            wavefunction = self.get_wavefunction(circuit).amplitudes
            n_qubits = len(circuit.qubits)

            # Pyquil does not support PauliSums with no terms.
            if len(qubit_operator.terms) == 0:
                return ExpectationValues(np.zeros((0,)))

            values = []

            for pauli_term in qubit_operator:
                sparse_pauli_term_ndarray = get_sparse_operator(
                    pauli_term, n_qubits=n_qubits
                ).toarray()
                if np.size(sparse_pauli_term_ndarray) == 1:
                    expectation_value = sparse_pauli_term_ndarray[0][0]
                    values.append(expectation_value)
                else:
                    expectation_value = np.real(
                        wavefunction.conj().T @ sparse_pauli_term_ndarray @ wavefunction
                    )
                    values.append(expectation_value)

            return expectation_values_to_real(ExpectationValues(np.asarray(values)))

    def get_exact_noisy_expectation_values(self, circuit, qubit_operator, **kwargs):
        """Run a circuit to prepare a wavefunction and measure the exact
        expectation values with respect to a given operator.
        Args:
            circuit (zquantum.core.circuit.Circuit): the circuit to prepare the state
            qubit_operator (openfermion.ops.QubitOperator): the operator to measure
        Returns:
            zquantum.core.measurement.ExpectationValues: the expectation values
                of each term in the operator
        """
        if self.noise_model is None:
            raise RuntimeError(
                "Please provide noise model to get exact noisy expectation values"
            )
        else:
            cirq_circuit = circuit.to_cirq()
            values = []
            n_qubits = len(circuit.qubits)
            for pauli_term in qubit_operator:
                sparse_pauli_term_ndarray = get_sparse_operator(
                    pauli_term, n_qubits=n_qubits
                ).toarray()
                if np.size(sparse_pauli_term_ndarray) == 1:
                    expectation_value = sparse_pauli_term_ndarray[0][0]
                    values.append(expectation_value)
                else:
                    noisy_circuit = cirq_circuit.with_noise(self.noise_model)
                    rho = self.simulator.simulate(noisy_circuit).final_density_matrix
                    expectation_value = np.real(
                        np.trace(rho @ sparse_pauli_term_ndarray)
                    )
                    values.append(expectation_value)
        return expectation_values_to_real(ExpectationValues(np.asarray(values)))

    def get_wavefunction(self, circuit):
        """Run a circuit and get the wavefunction of the resulting statevector.
        Args:
            circuit (zquantum.core.circuit.Circuit): the circuit to prepare the state
        Returns:
            wavefunction (pyquil.wavefuntion.Wavefunction): The wavefunction representing the circuit
        """
        super().get_wavefunction(circuit)

        amplitudes = circuit.to_cirq().final_state_vector()
        wavefunction = flip_wavefunction(Wavefunction(amplitudes))

        return wavefunction


def get_measurement_from_cirq_result_object(result_object, qubits):
    """Gets measurement bit strings from cirq result object and returns a Measurement object

    Args:


    Return:
        measurment (zquantum.core.measurement.Measurements)

    """

    keys = list(range(len(qubits)))

    numpy_samples = []
    for sub_key in keys:
        if sub_key in result_object._measurements.keys():
            numpy_samples.append(zip(*(result_object._measurements[str(sub_key)])))

    samples = []
    for numpy_bitstring in numpy_samples:
        bitstrings = []
        for key in numpy_bitstring:
            bitstrings.append(key[0])
        samples.append(tuple(bitstrings))

    measurement = Measurements()
    measurement.bitstrings = samples

    return measurement


def flip_wavefunction(wavefunction: Wavefunction):
    number_of_states = len(wavefunction.amplitudes)
    flipped_amplitudes = [None] * number_of_states
    for index in range(number_of_states):
        flipped_index = int(
            "0b" + bin(int(index))[2:].zfill(int(np.log2(number_of_states)))[::-1], 2
        )
        flipped_amplitudes[flipped_index] = wavefunction.amplitudes[index]
    return Wavefunction(flipped_amplitudes)
