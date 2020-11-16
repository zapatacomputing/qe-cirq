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

        measurements = self.simulator.run_circuit_and_measure(circuit)
        self.assertEqual(len(measurements.bitstrings), self.n_samples)

        for measurement in measurements.bitstrings:
            self.assertEqual(measurement, (1, 0, 0))
    

    def test_run_circuitset_and_measure(self):

            #Given
            circuit = Circuit(Program(X(0), CNOT(1, 2)))
            n_circuits = 5
            # When
            measurements_set = self.simulator.run_circuitset_and_measure([circuit] * n_circuits)
            # Then
            self.assertEqual(len(measurements_set), n_circuits)
            for measurements in measurements_set:
                self.assertEqual(len(measurements.bitstrings), self.n_samples)
                for measurement in measurements.bitstrings:
                    self.assertEqual(measurement, (1, 0, 0))


        

   