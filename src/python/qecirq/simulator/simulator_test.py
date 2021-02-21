import numpy as np
import pytest
from cirq import depolarize
from pyquil import Program
from pyquil.gates import X, H, CNOT
from openfermion.ops import QubitOperator

from zquantum.core.circuit import Circuit
from zquantum.core.interfaces.backend_test import (
    QuantumSimulatorTests,
    QuantumSimulatorGatesTest,
)

from .simulator import CirqSimulator


@pytest.fixture(
    params=[
        {
            "n_samples": 1,
        },
    ]
)
def backend(request):
    return CirqSimulator(**request.param)


@pytest.fixture()
def wf_simulator(request):
    return CirqSimulator()


@pytest.fixture()
def sampling_simulator(request):
    return CirqSimulator()


class TestCirqSimulator(QuantumSimulatorTests):
    def test_setup_basic_simulators(self):
        simulator = CirqSimulator()
        assert isinstance(simulator, CirqSimulator)
        assert simulator.n_samples is None
        assert simulator.noise_model is None

    def test_run_circuit_and_measure(self):
        # Given
        circuit = Circuit(Program(X(0), CNOT(1, 2)))
        simulator = CirqSimulator(n_samples=100)
        measurements = simulator.run_circuit_and_measure(circuit)
        assert len(measurements.bitstrings) == 100

        for measurement in measurements.bitstrings:
            assert measurement == (1, 0, 0)

    def test_run_circuitset_and_measure(self):

        # Given
        simulator = CirqSimulator(n_samples=100)
        circuit = Circuit(Program(X(0), CNOT(1, 2)))
        n_circuits = 5
        n_samples = 100
        # When
        measurements_set = simulator.run_circuitset_and_measure([circuit] * n_circuits)
        # Then
        assert len(measurements_set) == n_circuits
        for measurements in measurements_set:
            assert len(measurements.bitstrings) == n_samples
            for measurement in measurements.bitstrings:
                assert measurement == (1, 0, 0)

    def test_get_wavefunction(self):
        # Given
        simulator = CirqSimulator(n_samples=100)
        circuit = Circuit(Program(H(0), CNOT(0, 1), CNOT(1, 2)))

        # When
        wavefunction = simulator.get_wavefunction(circuit)
        # Then
        assert isinstance(wavefunction.amplitudes, np.ndarray)
        assert len(wavefunction.amplitudes) == 8
        assert np.isclose(
            wavefunction.amplitudes[0], (1 / np.sqrt(2) + 0j), atol=10e-15
        )
        assert np.isclose(
            wavefunction.amplitudes[7], (1 / np.sqrt(2) + 0j), atol=10e-15
        )

    def test_get_exact_expectation_values(self):
        # Given
        n_samples = 100
        simulator = CirqSimulator(n_samples=n_samples)
        circuit = Circuit(Program(H(0), CNOT(0, 1), CNOT(1, 2)))
        qubit_operator = QubitOperator("2[] - [Z0 Z1] + [X0 X2]")
        target_values = np.array([2.0, -1.0, 0.0])

        # When

        expectation_values = simulator.get_exact_expectation_values(
            circuit, qubit_operator
        )
        # Then
        np.testing.assert_array_almost_equal(expectation_values.values, target_values)

    def test_get_noisy_exact_expectation_values(self):
        # Given
        noise = 0.0002
        noise_model = depolarize(p=noise)
        simulator = CirqSimulator(noise_model=noise_model)
        circuit = Circuit(Program(H(0), CNOT(0, 1), CNOT(1, 2)))
        qubit_operator = QubitOperator("-[Z0 Z1] + [X0 X2]")
        target_values = np.array([-0.9986673775881747, 0.0])

        expectation_values = simulator.get_exact_noisy_expectation_values(
            circuit, qubit_operator
        )
        assert expectation_values.values[0] == target_values[0]
        assert expectation_values.values[1] == target_values[1]


class TestCirqSimulatorGates(QuantumSimulatorGatesTest):
    pass
