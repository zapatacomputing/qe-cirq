import unittest
import numpy as np
import os
from pyquil import Program
from pyquil.gates import H, CNOT, RX, CZ, X
from openfermion.ops import QubitOperator


from zquantum.core.circuit import Circuit
from zquantum.core.interfaces.backend_test import QuantumSimulatorTests
from zquantum.core.measurement import ExpectationValues
from ..simulator import CirqSimulator



class TestCirqSimulator(unittest.TestCase):
    def setUp(self):
        self.simulator = CirqSimulator(n_samples=10)
        # self.wf_simulator = CirqSimulator("statevector_simulator")
        # self.sampling_simulators = [CirqSimulator("qasm_simulator")]

        # self.noisy_simulators = [
        #     CirqSimulator(
        #         "qasm_simulator",
        #         n_samples=1000,
        #         noise_model=noise_model
        #     ),
        # ]
        # self.all_simulators = (
        #     [self.wf_simulator] + self.sampling_simulators + self.noisy_simulators
        # )

        # # Inherited tests
        # # NOTE: With our current version of qiskit, the statevector_simulator has a bug which won't return
        # #       the correct number of measurements
        # self.backends = self.sampling_simulators
        # self.wf_simulators = [self.wf_simulator]

    
    def test_run_circuit_and_measure(self):
        # Given
        circuit = Circuit(Program(X(0), CNOT(1, 2)))

        measurements = self.simulator.run_circuit_and_measure(circuit)
        self.assertEqual(len(measurements.bitstrings), 10)

        for measurement in measurements.bitstrings:
            self.assertEqual(measurement, (1, 0, 0))
            self.assertEqual(1,2)


    # def test_run_circuitset_and_measure(self):

    #         #Given
    #         circuit = Circuit(Program(X(0), CNOT(1, 2)))
    #         # When
    #         measurements_set = self.simulator.run_circuitset_and_measure([circuit] * 5)
    #         # Then
    #         self.assertEqual(len(measurements_set), 5)
    #         for measurements in measurements_set:
    #             self.assertEqual(len(measurements.bitstrings), 100)
    #             for measurement in measurements.bitstrings:
    #                 self.assertEqual(measurement, (1, 0, 0))

    # def test_setup_basic_simulators(self):
    #     simulator = CirqSimulator("qasm_simulator")
    #     self.assertIsInstance(simulator, CirqSimulator)
    #     self.assertEqual(simulator.device_name, "qasm_simulator")
    #     self.assertEqual(simulator.n_samples, None)
    #     self.assertEqual(simulator.noise_model, None)


    #     simulator = CirqSimulator("statevector_simulator")
    #     self.assertIsInstance(simulator, CirqSimulator)
    #     self.assertEqual(simulator.device_name, "statevector_simulator")
    #     self.assertEqual(simulator.n_samples, None)
    #     self.assertEqual(simulator.noise_model, None)
        

    # def test_expectation_value_with_noisy_simulator(self):
    #     for simulator in self.noisy_simulators:
    #         # Given
    #         # Initialize in |1> state
    #         circuit = Circuit(Program(X(0)))
    #         # Flip qubit an even number of times to remain in the |1> state, but allow decoherence to take effect
    #         circuit += Circuit(Program([X(0) for _ in range(10)]))
    #         qubit_operator = QubitOperator("Z0")
    #         simulator.n_samples = 8192
    #         # When
    #         expectation_values_10_gates = simulator.get_expectation_values(
    #             circuit, qubit_operator
    #         )
    #         # Then
    #         self.assertIsInstance(expectation_values_10_gates, ExpectationValues)
    #         self.assertEqual(len(expectation_values_10_gates.values), 1)
    #         self.assertGreater(expectation_values_10_gates.values[0], -1)
    #         self.assertLess(expectation_values_10_gates.values[0], 0.0)
    #         self.assertIsInstance(simulator, CirqSimulator)
    #         self.assertEqual(simulator.n_samples, 8192)
    #         self.assertIsInstance(simulator.noise_model, AerNoise.NoiseModel) # Change Test for noise model


    #         # Given
    #         # Initialize in |1> state
    #         circuit = Circuit(Program(X(0)))
    #         # Flip qubit an even number of times to remain in the |1> state, but allow decoherence to take effect
    #         circuit += Circuit(Program([X(0) for _ in range(100)]))
    #         qubit_operator = QubitOperator("Z0")
    #         simulator.n_samples = 8192
    #         # When
    #         expectation_values_100_gates = simulator.get_expectation_values(
    #             circuit, qubit_operator
    #         )
    #         # Then
    #         self.assertIsInstance(expectation_values_100_gates, ExpectationValues)
    #         self.assertEqual(len(expectation_values_100_gates.values), 1)
    #         self.assertGreater(expectation_values_100_gates.values[0], -1)
    #         self.assertLess(expectation_values_100_gates.values[0], 0.0)
    #         self.assertGreater(
    #             expectation_values_100_gates.values[0],
    #             expectation_values_10_gates.values[0],
    #         )
    #         self.assertIsInstance(simulator, QiskitSimulator)
    #         self.assertEqual(simulator.n_samples, 8192)
    #         self.assertIsInstance(simulator.noise_model, AerNoise.NoiseModel) # Change Test for noise model
