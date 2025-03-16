from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import PauliList
from qiskit.circuit import ParameterVector
from typing import List
import threading
import sys

from Code.VQLS.Ansatz import fixedAnsatz, controlledFixedAnsatz
from Code.VQLS.LCU import convertMatrixIntoCircuit
from Code.VQLS.LabelVector import controlledLabelVectorCircuit
from Code.Utils import getTotalAnsatzParameters, splitParameters, prepareBackend


def prepareCircuits(
    paulis: PauliList,
    bVector: List[float],
    qubits: int,
    isQuantumSimulation: bool,
    layers: int,
    threads: int,
    jobs: int,
    threading = False,
) -> tuple[List[QuantumCircuit], ParameterVector, List[QuantumCircuit], ParameterVector]:
    """
    Prepares the quantum circuits and parameter vectors for Hadamard tests and special Hadamard tests.

    Parameters:
    - paulis (PauliList): List of Pauli operators for the Hadamard test.
    - bVector (List[float]): Vector used to configure the label vector circuit.
    - qubits (int): Number of qubits in the quantum circuits.
    - isQuantumSimulation (bool): Flag to determine if the circuits are for quantum simulation.
    - layers (int): Number of layers in the ansatz circuits.
    - threads (int): Number of threads to use for parallel backend preparation.
    - jobs (int): Number of jobs to run in parallel.
    - threading (bool, optional): Flag to determine if threading should be used for parameter vector preparation. Defaults to False.

    Returns:
    - Tuple: A tuple containing:
        - List[QuantumCircuit]: List of Hadamard test circuits.
        - ParameterVector: Parameter vector for Hadamard test circuits.
        - List[QuantumCircuit]: List of special Hadamard test circuits.
        - ParameterVector: Parameter vector for special Hadamard test circuits.

    Notes:
    - If `threading` is True, parameter vector preparation is done in parallel threads.
    - If `threading` is False, parameter vectors are prepared sequentially.
    - Constructs circuits for label vector, fixed ansatz, and controlled fixed ansatz, and prepares Hadamard test circuits.
    """
    backend = prepareBackend(threads, jobs)
    if threading:
        # Prepare parameter vectors in parallel threads
        parameterVectorThread = ReturnValueThread(target=lambda: prepareParameterVector(
            "parametersHadarmard", qubits, layers
        ))

        parameterSpecialVectorThread = ReturnValueThread(target=lambda: prepareParameterVector(
            "parametersSpecialHadamard", qubits, layers))


        parameterVectorThread.start()
        parameterSpecialVectorThread.start()

        parametersHadamard, parametersHadamardSplit = parameterVectorThread.join()
        parametersSpecialHadamard, parametersSpecialHadamardSplit = parameterSpecialVectorThread.join()
    else:
        # Prepare parameter vectors sequentially
        parametersHadamard, parametersHadamardSplit = prepareParameterVector(
            "parametersHadarmard", qubits, layers
        )
        print("parametersHadamard prepared")
        parametersSpecialHadamard, parametersSpecialHadamardSplit = prepareParameterVector(
            "parametersSpecialHadamard", qubits, layers
        )

    print("parametersSpecialHadamard prepared")

    # Prepare label vector circuit
    labelVectorCircuit = QuantumCircuit(qubits + 1)
    controlledLabelVectorCircuit(labelVectorCircuit, 0, qubits, bVector)

    print("labelVectorCircuit prepared")
    # Prepare fixed ansatz circuit
    fixedAnsatzCircuit = QuantumCircuit(qubits + 1)
    fixedAnsatz(fixedAnsatzCircuit, qubits, parametersHadamardSplit, offset=1)
    print("fixedAnsatzCircuit prepared")

    # Prepare controlled fixed ansatz circuit
    controlledFixedAnsatzCircuit = QuantumCircuit(qubits + 1)
    controlledFixedAnsatz(
        controlledFixedAnsatzCircuit, qubits, parametersSpecialHadamardSplit
    )
    print("controlledFixedAnsatzCircuit prepared")
    if threading:
        # Prepare Hadamard test circuits in parallel threads
        hadamardCircuitsThread = ReturnValueThread(target=lambda: prepareHadamardTestCircuits(
            paulis, fixedAnsatzCircuit, qubits, isQuantumSimulation, backend
        ))

        specialHadamardCircuitsThread = ReturnValueThread(target=lambda: prepareSpecialHadamardTestCircuits(
            paulis, controlledFixedAnsatzCircuit, labelVectorCircuit, qubits, isQuantumSimulation, backend
        ))

        hadamardCircuitsThread.start()
        specialHadamardCircuitsThread.start()
        transpiledHadamardCircuits = hadamardCircuitsThread.join()
        transpiledSpecialHadamardCircuits = specialHadamardCircuitsThread.join()
    else:
        # Prepare Hadamard test circuits sequentially
        transpiledHadamardCircuits = prepareHadamardTestCircuits(paulis, fixedAnsatzCircuit, qubits, isQuantumSimulation, backend)
        print("transpiledHadamardCircuits prepared")
        transpiledSpecialHadamardCircuits = prepareSpecialHadamardTestCircuits(paulis, controlledFixedAnsatzCircuit, labelVectorCircuit, qubits, isQuantumSimulation, backend)
        print("transpiledSpecialHadamardCircuits prepared")

    return (
        transpiledHadamardCircuits,
        parametersHadamard,
        transpiledSpecialHadamardCircuits,
        parametersSpecialHadamard,
    )


def prepareHadamardTestCircuits(paulis, fixedAnsatzCircuit, qubits, isQuantumSimulation, backend) -> List[QuantumCircuit]:
    """
    Prepare and transpile a list of Hadamard test circuits for given Pauli operators.

    Parameters:
    - paulis (List[Pauli]): A list of Pauli operators used for Hadamard test circuits.
    - fixedAnsatzCircuit (QuantumCircuit): A fixed ansatz quantum circuit to be used in the Hadamard test.
    - qubits (int): Number of qubits in the quantum circuit.
    - isQuantumSimulation (bool): Flag indicating whether the backend is a quantum simulator.
    - backend: Backend to transpile the circuits for.

    Returns:
    - List[QuantumCircuit]: A list of transpiled Hadamard test circuits.

    Notes:
    - The function generates Hadamard test circuits for all pairs of Pauli operators.
    - Each pair of Pauli operators results in a separate Hadamard test circuit.
    - The generated circuits are transpiled for the specified backend.
    - The generated circuits are triangle matrix since:

         X X X X X

         . X X X X

         . . X X X

         . . . X X

         . . . . X

         lower triangular matrix is calculated using this formula: <0|V(a)^d A_n^d A_m V(a)|0> = (<0|V(a)^d A_m^d A_n V(a)|0>) conjugate
         c_n conj c_m <0|V(a)^d A_n^d A_m V(a)|0> = ( c_n conj c_m <0|V(a)^d A_m^d A_n V(a)|0>) conjugate
    """
    transpiledHadamardCircuits: List[QuantumCircuit] = []
    for i in range(len(paulis)):
        tempHadamardCircuits: List[QuantumCircuit] = []
        for j in range(i, len(paulis)):
            # Create Hadamard test wrapper to pass parameters to the construct circuit function
            def hadamardTestWrapper(circuit): return hadamardTest(
                circuit, [paulis[i], paulis[j]], fixedAnsatzCircuit
            )
            circ = constructCircuit(isQuantumSimulation, qubits + 1, hadamardTestWrapper)
            tempHadamardCircuits.append(circ)
        hadamardCircuits = transpile(tempHadamardCircuits, backend=backend)
        transpiledHadamardCircuits.extend(hadamardCircuits)
    return transpiledHadamardCircuits


def prepareSpecialHadamardTestCircuits(paulis, controlledFixedAnsatzCircuit, labelVectorCircuit, qubits, isQuantumSimulation, backend) -> List[QuantumCircuit]:
    """
    Prepare and transpile a list of special Hadamard test circuits for given Pauli operators.

    Parameters:
    - paulis (List[Pauli]): A list of Pauli operators used for special Hadamard test circuits.
    - controlledFixedAnsatzCircuit (QuantumCircuit): A fixed ansatz quantum circuit used for controlled Hadamard tests.
    - labelVectorCircuit (QuantumCircuit): A quantum circuit representing the label vector.
    - qubits (int): Number of qubits in the quantum circuit.
    - isQuantumSimulation (bool): Flag indicating whether the backend is a quantum simulator.
    - backend: Backend to transpile the circuits for.

    Returns:
    - List[QuantumCircuit]: A list of transpiled special Hadamard test circuits.

    Notes:
    - The function generates special Hadamard test circuits for each Pauli operator.
    - Each Pauli operator results in a separate special Hadamard test circuit.
    - The generated circuits are transpiled for the specified backend.
    """
    specialHadamardCircuits: List[QuantumCircuit] = []
    for i in range(len(paulis)):
        # Create special Hadamard test wrapper to pass parameters to the construct circuit function
        def specHadamardTestWrapper(circuit): return specialHadamardTest(
            circuit,
            [paulis[i]],
            controlledFixedAnsatzCircuit,
            labelVectorCircuit,
        )

        circ = constructCircuit(isQuantumSimulation,
                                qubits + 2, specHadamardTestWrapper)
        specialHadamardCircuits.append(circ)

    transpiledSpecialHadamardCircuits = transpile(
            specialHadamardCircuits, backend=backend
        )
    return transpiledSpecialHadamardCircuits


def getSolutionVector(circ: QuantumCircuit, qubits: int, outF: list):
    """
    Executes a quantum circuit to obtain the state vector.

    Parameters:
    - circ (QuantumCircuit): The quantum circuit to be executed.
    - qubits (int): The number of qubits used in the circuit.
    - outF (list): A list of integers used to modify the circuit with the `fixedAnsatz` function.

    Returns:
    - List[float]: The state vector of the quantum circuit after execution.

    Notes:
    - The function applies a fixed ansatz to the circuit, saves the state vector, and then runs the circuit using the Aer's simulator backend.
    """
    fixedAnsatz(circ, qubits, outF)
    circ.save_statevector()

    backend = Aer.get_backend("aer_simulator")

    t_circ = transpile(circ, backend)
    job = backend.run(t_circ)

    result = job.result()
    return result.get_statevector(circ, decimals=10)


def hadamardTest(
    circ: QuantumCircuit,
    paulis: PauliList,
    fixedAnsatzCircuit: QuantumCircuit,
):
    """
    Constructs a Hadamard test circuit for a given set of Pauli operators.

    Parameters:
    - circ (QuantumCircuit): The quantum circuit to be modified.
    - paulis (List[Pauli]): List of Pauli operators to be applied in the test.
    - fixedAnsatzCircuit (QuantumCircuit): A fixed ansatz quantum circuit to be included in the Hadamard test.

    Notes:
    - The function adds a Hadamard gate to an auxiliary qubit, applies the fixed ansatz circuit, converts Pauli matrices into quantum gates, and applies the Hadamard gate again.
    """
    auxiliaryIndex = 0
    circ.h(auxiliaryIndex)

    circ.barrier()

    circ.append(fixedAnsatzCircuit, range(fixedAnsatzCircuit.num_qubits))

    circ.barrier()

    convertMatrixIntoCircuit(
        circ,
        paulis,
        controlled=True,
        auxiliaryQubit=auxiliaryIndex,
        showBarriers=False,
    )

    circ.barrier()

    circ.h(auxiliaryIndex)


def specialHadamardTest(
    circ: QuantumCircuit,
    paulis: PauliList,
    controlledFixedAnsatzCircuit: QuantumCircuit,
    controlLabelVectorCircuit: QuantumCircuit,
):
    """
    Constructs a special Hadamard test circuit with additional control operations.

    Parameters:
    - circ (QuantumCircuit): The quantum circuit to be modified.
    - paulis (List[Pauli]): List of Pauli operators to be applied in the test.
    - controlledFixedAnsatzCircuit (QuantumCircuit): A controlled fixed ansatz quantum circuit.
    - controlLabelVectorCircuit (QuantumCircuit): A quantum circuit representing the control label vector.

    Notes:
    - The function adds a Hadamard gate to an auxiliary qubit, applies the controlled fixed ansatz circuit, converts Pauli matrices into quantum gates, applies the control label vector circuit, and applies the Hadamard gate again.
    """
    auxiliaryIndex = 0
    circ.h(auxiliaryIndex)

    circ.barrier()

    circ.append(
        controlledFixedAnsatzCircuit, range(controlledFixedAnsatzCircuit.num_qubits)
    )

    circ.barrier()

    convertMatrixIntoCircuit(
        circ,
        paulis,
        controlled=True,
        auxiliaryQubit=auxiliaryIndex,
        showBarriers=False,
    )

    circ.barrier()

    circ.append(controlLabelVectorCircuit, range(controlLabelVectorCircuit.num_qubits))

    circ.barrier()

    circ.h(auxiliaryIndex)


def constructCircuit(
    isQuantumSimulation: bool, totalNeededQubits: int, circuitLambda: callable
) -> QuantumCircuit:
    """
    Constructs a quantum circuit based on whether it's for quantum simulation or not.

    Parameters:
    - isQuantumSimulation (bool): Flag to indicate if the circuit is for quantum simulation.
    - totalNeededQubits (int): The total number of qubits needed in the circuit.
    - circuitLambda (Callable[[QuantumCircuit], None]): A function to modify the circuit.

    Returns:
    - QuantumCircuit: The constructed and modified quantum circuit.

    Notes:
    - If `isQuantumSimulation` is True, the circuit is prepared for simulation with a measurement operation.
    - If False, the circuit is prepared for statevector output.
    """
    if isQuantumSimulation:
        circ: QuantumCircuit = QuantumCircuit(totalNeededQubits, 1)
        circuitLambda(circ)
        circ.measure(0, 0)
    else:
        circ: QuantumCircuit = QuantumCircuit(totalNeededQubits)
        circuitLambda(circ)
        circ.save_statevector()

    return circ


def prepareParameterVector(name: str, qubits: int, layers: int) -> tuple[ParameterVector, List[List[int]]]:
    """
    Prepares a parameter vector for use in a quantum circuit ansatz.

    Parameters:
    - name (str): Name for the parameter vector.
    - qubits (int): Number of qubits in the ansatz circuit.
    - layers (int): Number of layers in the ansatz circuit.

    Returns:
    - tuple: A tuple containing the parameter vector and a list of split parameters.

    Notes:
    - The function generates a parameter vector and splits it according to the number of qubits and layers in the ansatz circuit.
    """
    totalParamsNeeded: int = getTotalAnsatzParameters(qubits, layers)
    parameters: ParameterVector = ParameterVector(name, totalParamsNeeded)
    return parameters, splitParameters(parameters, qubits, alternating=qubits != 3)


class ReturnValueThread(threading.Thread):
    """
    A custom thread class that allows returning a result from the thread's target function.

    Inherits from `threading.Thread` and adds functionality to return the result of the thread's execution.

    Attributes:
    - result: The result of the thread's target function.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.result = None

    def run(self):
        """
        Executes the thread's target function and stores the result.

        Handles exceptions and prints any errors to stderr.
        """
        if self._target is None:
            return
        try:
            self.result = self._target(*self._args, **self._kwargs)
        except Exception as exc:
            print(f'{type(exc).__name__}: {exc}', file=sys.stderr)

    def join(self, *args, **kwargs):
        """
        Waits for the thread to complete and returns the result.

        Parameters:
        - *args: Variable length argument list passed to `join`.
        - **kwargs: Arbitrary keyword arguments passed to `join`.

        Returns:
        - The result of the thread's target function.
        """
        super().join(*args, **kwargs)
        return self.result