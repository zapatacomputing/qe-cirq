import numpy as np
from openfermion.ops import QubitOperator, IsingOperator
from openfermion.transforms import get_sparse_operator

from qeopenfermion import expectation, change_operator_type
from zquantum.core.interfaces.backend import QuantumSimulator
from zquantum.core.measurement import (
    expectation_values_to_real,
    ExpectationValues,
    Measurements,
)

from cirq import Circuit, measure, Simulator
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
        qubits = list(cirq_circuit.all_qubits())
        cirq_circuit.append(measure(*qubits, key= "result"))
        
        counts_dict = Simulator().run(cirq_circuit, repetitions= self.n_samples).histogram(key='result')
        print(counts_dict)
        counts_dict = {"{0:b}".format(intstring).zfill(num_qubits): counts_dict[intstring] for intstring in counts_dict}
        measurements = Measurements.from_counts(counts_dict)
        print('bitstrings: ', measurements.bitstrings)
        
        # reversed_counts_dict ={}
        # for bitstring in counts_dict.keys():
        #     reversed_counts_dict[bitstring[::-1]] = counts_dict[bitstring]
        return Measurements.from_counts(counts_dict)


    # def run_circuitset_and_measure(self, circuitset, **kwargs):
    #     """ Run a set of circuits and measure a certain number of bitstrings.
    #     Note: the number of bitstrings measured is derived from self.n_samples
    #     Args:
    #         circuit (zquantum.core.circuit.Circuit): the circuit to prepare the state
    #     Returns:
    #         a list of lists of bitstrings (a list of lists of tuples)
    #     """
    #     cirq_circuitset = []
    #     measurements_set = []
    #     for circuit in circuitset:
    #         num_qubits = len(circuit.qubits)
    #         cirq_circuit = circuit.to_cirq()
    #         print(cirq_circuit)
    #         qubits = list(cirq_circuit.all_qubits())
    #         cirq_circuit.append(measure(*qubits, key= "result"))
    #         counts_dict = Simulator().run(cirq_circuit, repetitions= self.n_samples).histogram(key="result")
    #         counts_dict = {"{0:b}".format(intstring).zfill(num_qubits): counts_dict[intstring] for intstring in counts_dict}
            
    #         print(counts_dict)
    #         reversed_counts_dict ={}
    #         # for bitstring in counts_dict.keys():
    #         #     reversed_counts_dict[bitstring[::-1]] = counts_dict[bitstring]
    #         # print('----------------------------------------------------------')
    #         # print(reversed_counts_dict)
    #         measurements = Measurements.from_counts(counts_dict)
    #         measurements_set.append(measurements)

    #     return measurements_set

    def get_expectation_values(self, circuit, qubit_operator, **kwargs):
        """ Run a circuit and measure the expectation values with respect to a 
        given operator. Note: the number of bitstrings measured is derived
        from self.n_samples - if self.n_samples = None, then this will use
        self.get_exact_expectation_values
        Args:
            circuit (zquantum.core.circuit.Circuit): the circuit to prepare the state
            qubit_operator (openfermion.ops.QubitOperator): the operator to measure
        Returns:
            zquantum.core.measurement.ExpectationValues: the expectation values
                of each term in the operator
        """
        self.num_circuits_run += 1
        self.num_jobs_run += 1
        if self.n_samples == None:
            return self.get_exact_expectation_values(circuit, qubit_operator, **kwargs)
        else:
            operator = change_operator_type(qubit_operator, IsingOperator)
            measurements = self.run_circuit_and_measure(circuit)
            expectation_values = measurements.get_expectation_values(operator)

            expectation_values = expectation_values_to_real(expectation_values)
            return expectation_values

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
        wavefunction = self.get_wavefunction(circuit)

        # Pyquil does not support PauliSums with no terms.
        if len(qubit_operator.terms) == 0:
            return ExpectationValues(np.zeros((0,)))

        values = []

        for pauli_term in qubit_operator:
            sparse_pauli_term_ndarray = get_sparse_operator(pauli_term).toarray()
            expectation_value = np.real(wavefunction.conj().T @ parse_pauli_term_ndarray @ wavefunction)
            values.append(expectation_value)
        return expectation_values_to_real(ExpectationValues(np.asarray(values)))

    def get_expectation_values_for_circuitset(self, circuitset, operator, **kwargs):
        """ Run a set of circuits and measure the expectation values with respect to a 
        given operator. 
        Args:
            circuitset (list of zquantum.core.circuit.Circuit objects): the circuits to prepare the states
            operator (openfermion.ops.IsingOperator or openfermion.ops.QubitOperator): the operator to measure
        Returns:
            list of zquantum.core.measurement.ExpectationValues objects: a list of the expectation values of each 
                term in the operator with respect to the various state preparation circuits
        """
        self.num_circuits_run += len(circuitset)
        self.num_jobs_run += 1
        operator = change_operator_type(operator, IsingOperator)
        measurements_set = self.run_circuitset_and_measure(circuitset)

        expectation_values_set = []
        for measurements in measurements_set:
            expectation_values = measurements.get_expectation_values(operator)
            expectation_values = expectation_values_to_real(expectation_values)
            expectation_values_set.append(expectation_values)

        return expectation_values_set

    def get_wavefunction(self, circuit):
        """ Run a circuit and get the wavefunction of the resulting statevector.
        Args:
            circuit (zquantum.core.circuit.Circuit): the circuit to prepare the state
        Returns:
            pyquil.wavefunction.Wavefunction. # TODO:change return type
        """

        wavefunction = circuit.to_cirq.final_wavefunction()
 
        return wavefunction
