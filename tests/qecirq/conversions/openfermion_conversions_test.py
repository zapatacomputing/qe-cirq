import unittest

from cirq import GridQubit, LineQubit, PauliString, PauliSum, X, Y, Z
from qecirq.conversions import qubitop_to_paulisum
from zquantum.core.openfermion import QubitOperator


class TestQubitOperator(unittest.TestCase):
    def test_qubitop_to_paulisum_identity_operator(self):
        # Given
        qubit_operator = QubitOperator("", 4)

        # When
        paulisum = qubitop_to_paulisum(qubit_operator)

        # Then
        self.assertEqual(paulisum.qubits, ())
        self.assertEqual(paulisum, PauliSum() + 4)

    def test_qubitop_to_paulisum_z0z1_operator(self):
        # Given
        qubit_operator = QubitOperator("Z0 Z1", -1.5)
        expected_qubits = (GridQubit(0, 0), GridQubit(1, 0))
        expected_paulisum = (
            PauliSum()
            + PauliString(Z.on(expected_qubits[0]))
            * PauliString(Z.on(expected_qubits[1]))
            * -1.5
        )

        # When
        paulisum = qubitop_to_paulisum(qubit_operator)

        # Then
        self.assertEqual(paulisum.qubits, expected_qubits)
        self.assertEqual(paulisum, expected_paulisum)

    def test_qubitop_to_paulisum_setting_qubits(self):
        # Given
        qubit_operator = QubitOperator("Z0 Z1", -1.5)
        expected_qubits = (LineQubit(0), LineQubit(5))
        expected_paulisum = (
            PauliSum()
            + PauliString(Z.on(expected_qubits[0]))
            * PauliString(Z.on(expected_qubits[1]))
            * -1.5
        )

        # When
        paulisum = qubitop_to_paulisum(qubit_operator, qubits=expected_qubits)

        # Then
        self.assertEqual(paulisum.qubits, expected_qubits)
        self.assertEqual(paulisum, expected_paulisum)

    def test_qubitop_to_paulisum_more_terms(self):
        # Given
        qubit_operator = (
            QubitOperator("Z0 Z1 Z2", -1.5)
            + QubitOperator("X0", 2.5)
            + QubitOperator("Y1", 3.5)
        )
        expected_qubits = (LineQubit(0), LineQubit(5), LineQubit(8))
        expected_paulisum = (
            PauliSum()
            + (
                PauliString(Z.on(expected_qubits[0]))
                * PauliString(Z.on(expected_qubits[1]))
                * PauliString(Z.on(expected_qubits[2]))
                * -1.5
            )
            + (PauliString(X.on(expected_qubits[0]) * 2.5))
            + (PauliString(Y.on(expected_qubits[1]) * 3.5))
        )

        # When
        paulisum = qubitop_to_paulisum(qubit_operator, qubits=expected_qubits)

        # Then
        self.assertEqual(paulisum.qubits, expected_qubits)
        self.assertEqual(paulisum, expected_paulisum)
