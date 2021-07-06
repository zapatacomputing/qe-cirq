import sys
from typing import List, Optional

import cirq
import numpy as np
from openfermion import get_sparse_operator
from pyquil.wavefunction import Wavefunction
from zquantum.core.circuits import Circuit, export_to_cirq
from zquantum.core.interfaces.backend import QuantumSimulator
from zquantum.core.measurement import (
    ExpectationValues,
    Measurements,
    expectation_values_to_real,
)


def _prepare_measurable_cirq_circuit(circuit, noise_model):
    """Export circuit to Cirq and add terminal measurements."""
    cirq_circuit = export_to_cirq(circuit)

    if noise_model is not None:
        cirq_circuit = cirq_circuit.with_noise(noise_model)

    cirq_circuit.append(cirq.measure_each(*cirq_circuit.all_qubits()))

    return cirq_circuit


class CirqSimulator(QuantumSimulator):
    """Simulator using a cirq device (simulator or QPU).

    Currently this Simulator uses cirq.Simulator if noise_model is None and
    cirq.DensityMatrixSimulator otherwise.

    Args:
        noise_model: an optional noise model to pass in for noisy simulations

    Attributes:
        noise_model: an optional noise model to pass in for noisy simulations
        simulator: Cirq simulator this class uses.
    """

    supports_batching = True
    batch_size = sys.maxsize

    def __init__(self, noise_model=None, seed=None):
        super().__init__()
        self.noise_model = noise_model
        if self.noise_model is not None:
            self.simulator = cirq.DensityMatrixSimulator(dtype=np.complex128, seed=seed)
        else:
            self.simulator = cirq.Simulator(seed=seed)

    def run_circuit_and_measure(self, circuit: Circuit, n_samples=None, **kwargs):
        """Run a circuit and measure a certain number of bitstrings.

        Args:
            circuit: the circuit to prepare the state.
            n_samples: number of bitstrings to measure. If None, `self.n_samples`
                is used.
        Returns:
            A list of bitstrings.
        """
        super().run_circuit_and_measure(circuit, n_samples)

        result_object = self.simulator.run(
            _prepare_measurable_cirq_circuit(circuit, self.noise_model),
            repetitions=n_samples,
        )

        measurement = get_measurement_from_cirq_result_object(
            result_object, circuit.n_qubits, n_samples
        )

        return measurement

    def run_circuitset_and_measure(
        self, circuitset: List[Circuit], n_samples: List[int], **kwargs
    ):
        """Run a set of circuits and measure a certain number of bitstrings.

        Args:
            circuitset: a set of circuits to prepare the state.
            n_samples: number of bitstrings to measure. If None, `self.n_samples`
                is used. If an iterable, its-ith element corresponds to number
                of samples that will be taken from i-th circuit. If an int N,
                each circuit in circuitset will be measured N times.
        Returns:
            a list of lists of bitstrings (a list of lists of tuples)
        """
        super().run_circuitset_and_measure(circuitset, n_samples)

        cirq_circuitset = [
            _prepare_measurable_cirq_circuit(circuit, self.noise_model)
            for circuit in circuitset
        ]

        result = self.simulator.run_batch(cirq_circuitset, repetitions=n_samples)

        measurements_set = [
            get_measurement_from_cirq_result_object(
                sub_result[0], circuit.n_qubits, num_samples
            )
            for sub_result, circuit, num_samples in zip(result, circuitset, n_samples)
        ]

        return measurements_set

    def get_exact_expectation_values(self, circuit: Circuit, qubit_operator, **kwargs):
        """Compute exact expectation values with respect to given operator.

        Args:
            circuit: the circuit to prepare the state
            qubit_operator: the operator to measure
        Returns:
            the expectation values of each term in the operator
        """
        if self.noise_model is not None:
            return self.get_exact_noisy_expectation_values(circuit, qubit_operator)
        else:
            wavefunction = self.get_wavefunction(circuit).amplitudes

            # Pyquil does not support PauliSums with no terms.
            if len(qubit_operator.terms) == 0:
                return ExpectationValues(np.zeros((0,)))

            values = []

            for pauli_term in qubit_operator:
                sparse_pauli_term_ndarray = get_sparse_operator(
                    pauli_term, n_qubits=circuit.n_qubits
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

    def get_exact_noisy_expectation_values(self, circuit: Circuit, qubit_operator):
        """Compute exact expectation values w.r.t. given operator in presence of noise.

        Note that this method can be used only if simulator's noise_model is not set
        to None.

        Args:
            circuit: the circuit to prepare the state
            qubit_operator: the operator to measure
        Returns:
            the expectation values of each term in the operator
        Raises:
            RuntimeError if this simulator's noise_model is None.
        """
        if self.noise_model is None:
            raise RuntimeError(
                "Please provide noise model to get exact noisy expectation values"
            )
        else:
            cirq_circuit = export_to_cirq(circuit)
            values = []

            for pauli_term in qubit_operator:
                sparse_pauli_term_ndarray = get_sparse_operator(
                    pauli_term, n_qubits=circuit.n_qubits
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

    def get_wavefunction(self, circuit: Circuit):
        """Run a circuit and get the wavefunction of the resulting statevector.

        Args:
            circuit: the circuit to prepare the state
        Returns:
            wavefunction: The wavefunction representing the final state of the circuit
        """
        super().get_wavefunction(circuit)

        amplitudes = export_to_cirq(circuit).final_state_vector()
        wavefunction = flip_wavefunction(Wavefunction(amplitudes))

        return wavefunction


def get_measurement_from_cirq_result_object(result_object, n_qubits, n_samples):
    """Extract measurement bitstrings from cirq result object.

    Args:
        result_object: object returned by Cirq simulator's run or run_batch.
        n_qubits: number of qubits in full circuit (before exporting to cirq).
        n_samples: number of measured samples
    Return:
        Measurements.
    """
    numpy_samples = list(
        zip(
            *(
                result_object.measurements.get(str(sub_key), [[0]] * n_samples)
                for sub_key in range(n_qubits)
            )
        )
    )

    samples = [
        tuple(key[0] for key in numpy_bitstring) for numpy_bitstring in numpy_samples
    ]

    measurement = Measurements(samples)
    return measurement


def _flip_bits(n, num_bits):
    return int(bin(n)[2:].zfill(num_bits)[::-1], 2)


def flip_wavefunction(wavefunction: Wavefunction):
    number_of_states = len(wavefunction.amplitudes)
    ordering = [
        _flip_bits(n, number_of_states.bit_length() - 1)
        for n in range(number_of_states)
    ]
    flipped_amplitudes = [wavefunction.amplitudes[i] for i in ordering]
    return Wavefunction(np.array(flipped_amplitudes))
