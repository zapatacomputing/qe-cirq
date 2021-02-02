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

from cirq import (depolarize, asymmetric_depolarize, 
                generalized_amplitude_damp, amplitude_damp,
                phase_damp, phase_flip, bit_flip)



class TestCirqSimulator(unittest.TestCase):
    def setUp(self):
        self.n_samples = 100
        self.simulator = CirqSimulator(n_samples=self.n_samples)

    def test_setup_basic_simulators(self):
        simulator = CirqSimulator()
        self.assertIsInstance(simulator, CirqSimulator)
        self.assertEqual(simulator.n_samples, None)
        self.assertEqual(simulator.noise_model, None)
    
    def test_run_circuit_and_measure(self):
        # Given
        circuit = Circuit(Program(X(0), CNOT(1, 2)))
        simulator = CirqSimulator(n_samples=self.n_samples)
        measurements = simulator.run_circuit_and_measure(circuit)
        self.assertEqual(len(measurements.bitstrings), self.n_samples)

        for measurement in measurements.bitstrings:
            self.assertEqual(measurement, (1, 0, 0))
    
    def test_run_circuitset_and_measure(self):

            #Given
            simulator = CirqSimulator(n_samples=self.n_samples)
            circuit = Circuit(Program(X(0), CNOT(1, 2)))
            n_circuits = 5
            # When
            measurements_set = simulator.run_circuitset_and_measure([circuit] * n_circuits)
            # Then
            self.assertEqual(len(measurements_set), n_circuits)
            for measurements in measurements_set:
                self.assertEqual(len(measurements.bitstrings), self.n_samples)
                for measurement in measurements.bitstrings:
                    self.assertEqual(measurement, (1, 0, 0))

    def test_get_wavefunction(self):
        # Given
        simulator = CirqSimulator(n_samples=self.n_samples)
        circuit = Circuit(Program(H(0), CNOT(0, 1), CNOT(1, 2)))

        # When
        wavefunction = simulator.get_wavefunction(circuit)
        # Then
        self.assertIsInstance(wavefunction, np.ndarray)
        self.assertEqual(len(wavefunction), 8)
        self.assertAlmostEqual(wavefunction[0], (1 / np.sqrt(2) + 0j))
        self.assertAlmostEqual(wavefunction[7], (1 / np.sqrt(2) + 0j))

    def test_get_exact_expectation_values(self):
        # Given
        simulator = CirqSimulator(n_samples=self.n_samples)
        circuit = Circuit(Program(H(0), CNOT(0, 1), CNOT(1, 2)))
        qubit_operator = QubitOperator("2[] - [Z0 Z1] + [X0 X2]")
        target_values = np.array([2.0, -1.0, 0.0])

        # When
      
        expectation_values = simulator.get_exact_expectation_values(
            circuit, qubit_operator
        )
        # Then
        np.testing.assert_array_almost_equal(
            expectation_values.values, target_values
        )
        self.assertIsInstance(expectation_values.values, np.ndarray)

    def test_get_noisy_exact_expectation_values(self):
        # Given
        noise = 0.00002
        noise_model = depolarize(p=noise)
        simulator = CirqSimulator(noise_model =noise_model)
        circuit = Circuit(Program(H(0), CNOT(0, 1), CNOT(1, 2)))
        qubit_operator = QubitOperator("-[Z0 Z1] + [X0 X2]")
        target_values = np.array([ -1.0, 0.0])

        expectation_values = simulator.get_exact_noisy_expectation_values(
            circuit, qubit_operator
        )
        self.assertEqual(expectation_values.values[0], target_values[0])

    


        

   