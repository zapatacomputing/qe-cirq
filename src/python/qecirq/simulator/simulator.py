from typing import Iterable
import numpy as np
from openfermion import get_sparse_operator

from zquantum.core.interfaces.backend import QuantumSimulator
from zquantum.core.measurement import (
    expectation_values_to_real,
    ExpectationValues,
    Measurements,
)

from cirq import Simulator, measure_each
from cirq import DensityMatrixSimulator
from pyquil.wavefunction import Wavefunction
import sys

from zquantum.core.circuit import Circuit as OldCircuit
from zquantum.core.wip.circuits import export_to_cirq, new_circuit_from_old_circuit
from zquantum.core.wip.compatibility_tools import compatible_with_old_type


def _prepare_runnable_cirq_circuit(circuit, noise_model):
    """Export circuit to Cirq and add terminal measurements."""
    cirq_circuit = export_to_cirq(circuit)
    if noise_model is not None:
        cirq_circuit = cirq_circuit.with_noise(noise_model)

    qubits = list(cirq_circuit.all_qubits())

    for i in range(0, len(qubits)):
        cirq_circuit.append(measure_each(qubits[i]))
    return cirq_circuit


class CirqSimulator(QuantumSimulator):
    """Simulator using a cirq device (simulator or QPU).

    Currently this Simulator uses cirq.Simulator if noise_model is None and
    cirq.DensityMatrixSimulator otherwise.

    Args:
        n_samples: the number of samples to use when running the device
        noise_model: an optional noise model to pass in for noisy simulations

    Attributes:
        n_samples: the number of samples to use when running the device
        noise_model: an optional noise model to pass in for noisy simulations
        simulator: Cirq simulator this class uses.
    """
    supports_batching = True
    batch_size = sys.maxsize

    def __init__(
        self,
        n_samples=None,
        noise_model=None,
    ):
        super().__init__(n_samples)
        self.noise_model = noise_model
        if self.noise_model is not None:
            self.simulator = DensityMatrixSimulator(dtype=np.complex128)
        else:
            self.simulator = Simulator()

    @compatible_with_old_type(
        old_type=OldCircuit, translate_old_to_wip=new_circuit_from_old_circuit
    )
    def run_circuit_and_measure(self, circuit, n_samples=None, **kwargs):
        """Run a circuit and measure a certain number of bitstrings.

        Args:
            circuit: the circuit to prepare the state.
            n_samples: number of bitstrings to measure. If None, `self.n_samples`
                is used.
        Returns:
            A list of bitstrings.
        """
        super().run_circuit_and_measure(circuit)
        if n_samples is None:
            n_samples = self.n_samples

        result_object = self.simulator.run(
            _prepare_runnable_cirq_circuit(circuit, self.noise_model),
            repetitions=n_samples
        )

        measurement = get_measurement_from_cirq_result_object(
            result_object, range(circuit.n_qubits), n_samples
        )

        return measurement

    @compatible_with_old_type(
        old_type=OldCircuit,
        translate_old_to_wip=new_circuit_from_old_circuit,
        consider_iterable_types=[list, tuple]
    )
    def run_circuitset_and_measure(self, circuitset, n_samples=None, **kwargs):
        """Run a set of circuits and measure a certain number of bitstrings.
        Note: the number of bitstrings measured is derived from self.n_samples
        Args:
            circuit (zquantum.core.circuit.Circuit): the circuit to prepare the state
        Returns:
            a list of lists of bitstrings (a list of lists of tuples)
        """
        super().run_circuitset_and_measure(circuitset)

        if n_samples is None and self.n_samples is None:
            raise ValueError(
                "The n_samples passed to run_circuitset_and_measure and simulator's "
                "default n_samples cannot be None at the same time"
            )
        if n_samples is None:
            n_samples = [self.n_samples for circuit in circuitset]
        if not isinstance(n_samples, Iterable):
            n_samples = [n_samples] * len(circuitset)

        cirq_circuitset = [
            _prepare_runnable_cirq_circuit(circuit, self.noise_model)
            for circuit in circuitset
        ]
        measurements_set = []

        result = self.simulator.run_batch(cirq_circuitset, repetitions=n_samples)

        for i in range(len(cirq_circuitset)):
            measurements = get_measurement_from_cirq_result_object(
                result[i][0], range(circuitset[i].n_qubits), n_samples[i]
            )
            measurements_set.append(measurements)

        return measurements_set

    @compatible_with_old_type(
        old_type=OldCircuit, translate_old_to_wip=new_circuit_from_old_circuit
    )
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

    @compatible_with_old_type(
        old_type=OldCircuit, translate_old_to_wip=new_circuit_from_old_circuit
    )
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

    @compatible_with_old_type(
        old_type=OldCircuit, translate_old_to_wip=new_circuit_from_old_circuit
    )
    def get_wavefunction(self, circuit):
        """Run a circuit and get the wavefunction of the resulting statevector.
        Args:
            circuit (zquantum.core.circuit.Circuit): the circuit to prepare the state
        Returns:
            wavefunction (pyquil.wavefuntion.Wavefunction): The wavefunction representing the circuit
        """
        super().get_wavefunction(circuit)

        amplitudes = export_to_cirq(circuit).final_state_vector()
        wavefunction = flip_wavefunction(Wavefunction(amplitudes))

        return wavefunction


def get_measurement_from_cirq_result_object(result_object, qubits, n_samples):
    """Gets measurement bit strings from cirq result object and returns a Measurement object

    Args:


    Return:
        measurment (zquantum.core.measurement.Measurements)

    """

    keys = list(range(len(qubits)))

    numpy_samples = list(
        zip(
            *(
                result_object._measurements.get(str(sub_key), [[0]] * n_samples)
                for sub_key in keys
            )
        )
    )

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
