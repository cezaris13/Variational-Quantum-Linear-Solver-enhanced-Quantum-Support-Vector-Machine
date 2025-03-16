from typing import List
from qiskit import QuantumCircuit


def fixedAnsatz(
    circ: QuantumCircuit,
    qubits: int,
    parameters: List[List[float]],
    offset: int = 0,
    layers: int = 3,
    barrier: bool = False,
):
    """
    Applies a fixed ansatz to the provided quantum circuit.

    Parameters:
    - circ (QuantumCircuit): The quantum circuit to which the ansatz is applied.
    - qubits (int): Number of qubits in the circuit.
    - parameters (List[List[float]]): List of parameters for the ansatz gates.
    - offset (int, optional): Offset for qubit indices. Defaults to 0.
    - layers (int, optional): Number of layers in the ansatz. Defaults to 3.
    - barrier (bool, optional): Whether to add barriers between different gate types. Defaults to False.

    Returns:
    - None
    """
    gates = getFixedAnsatzGates(qubits, parameters, offset=offset, layers=layers)
    gatesToCircuit(circ, gates, barrier=barrier)


def controlledFixedAnsatz(
    circ: QuantumCircuit,
    qubits: int,
    parameters: List[List[float]],
    barrier: bool = False,
):
    """
    Applies a controlled fixed ansatz to the provided quantum circuit for the Hadamard test.

    Parameters:
    - circ (QuantumCircuit): The quantum circuit to which the ansatz is applied.
    - qubits (int): Number of qubits in the circuit.
    - parameters (List[List[float]]): List of parameters for the ansatz gates.
    - barrier (bool, optional): Whether to add barriers between different gate types. Defaults to False.

    Note:
    - Creates controlled anstaz for calculating |<b|psi>|^2 with a Hadamard test
    Returns:
    - None
    """
    gates = getControlledFixedAnsatzGates(qubits, parameters)
    gatesToCircuit(circ, gates, barrier=barrier)


def gatesToCircuit(circuit: QuantumCircuit, gateList: tuple[str, tuple[float, float]], barrier: bool = False):
    """
    Applies a list of gates to a quantum circuit.

    Parameters:
    - circuit (QuantumCircuit): The quantum circuit to which the gates are applied.
    - gateList: List of gates to apply, where each gate is represented as a tuple (gate_type, parameters).
    - barrier (bool, optional): Whether to add barriers between different gate types. Defaults to False.

    Returns:
    - None
    """
    lastGate: str = ""
    for i in range(len(gateList)):
        if lastGate != gateList[i][0] and barrier:
            circuit.barrier()
        lastGate = gateList[i][0]
        if gateList[i][0] == "Ry":  # ("Ry", (theta, qubit))
            circuit.ry(gateList[i][1][0], gateList[i][1][1])
        elif gateList[i][0] == "Rx":  # ("Rx", (theta, qubit))
            circuit.rx(gateList[i][1][0], gateList[i][1][1])
        elif gateList[i][0] == "Rz":  # ("Rz", (theta, qubit))
            circuit.rz(gateList[i][1][0], gateList[i][1][1])
        elif gateList[i][0] == "CNOT":  # ("CNOT", (control, target))
            circuit.cx(gateList[i][1][0], gateList[i][1][1])
        elif gateList[i][0] == "CZ":  # ("CZ", (control, target))
            circuit.cz(gateList[i][1][0], gateList[i][1][1])
        elif gateList[i][0] == "CCNOT":  # ("CCNOT", (control1, control2, target))
            circuit.ccx(gateList[i][1][0], gateList[i][1][1], gateList[i][1][2])
        elif gateList[i][0] == "CCZ":  # ("CCZ", (control1, control2, target))
            circuit.ccz(gateList[i][1][0], gateList[i][1][1], gateList[i][1][2])
        elif gateList[i][0] == "CRx":  # ("CRx", (theta, (control, target))
            circuit.crx(gateList[i][1][0], gateList[i][1][1][0], gateList[i][1][1][1])
        elif gateList[i][0] == "CRy":  # ("CRy", (theta, (control, target))
            circuit.cry(gateList[i][1][0], gateList[i][1][1][0], gateList[i][1][1][1])
        elif gateList[i][0] == "CRz":  # ("CRz", (theta, (control, target))
            circuit.crz(gateList[i][1][0], gateList[i][1][1][0], gateList[i][1][1][1])


def getFixedAnsatzGates(
    qubits: int, parameters: List[List[float]], offset: int = 0, layers: int = 3
) -> tuple[str, tuple[float, float]]:
    """
    Generates a list of gates for the fixed ansatz based on the number of qubits, parameters, and layers.

    Parameters:
    - qubits (int): Number of qubits in the circuit.
    - parameters (List[List[float]]): List of parameters for the ansatz gates.
    - offset (int, optional): Offset for qubit indices. Defaults to 0.
    - layers (int, optional): Number of layers in the ansatz. Defaults to 3.

    Returns:
    - List of gates as tuples, where each tuple is (gate_type, parameters).

    Raises:
    - Exception: If the number of qubits is less than 3.
    """
    if qubits < 3:
        raise Exception("Qubits must be at least 3")

    gates = []
    qubitList = [i + offset for i in range(qubits)]
    if qubits == 3:
        # https://arxiv.org/pdf/1909.05820 page 20
        for i in range(qubits):
            gates.append(("Ry", (parameters[0][i], qubitList[i])))
        gates.append(("CZ", (qubitList[0], qubitList[1])))
        gates.append(("CZ", (qubitList[2], qubitList[0])))

        for i in range(qubits):
            gates.append(("Ry", (parameters[1][i], qubitList[i])))
        gates.append(("CZ", (qubitList[1], qubitList[2])))
        gates.append(("CZ", (qubitList[2], qubitList[0])))

        for i in range(qubits):
            gates.append(("Ry", (parameters[2][i], qubitList[i])))
    else:
        # https://arxiv.org/pdf/1909.05820 page 5
        for i in range(qubits):
            gates.append(("Ry", (parameters[0][i], qubitList[i])))

        layer = 1
        for i in range(layers):
            for i in range(qubits // 2):
                gates.append(("CZ", (qubitList[2 * i], qubitList[2 * i + 1])))
            for i in range(qubits):
                gates.append(("Ry", (parameters[layer][i], qubitList[i])))
            layer = layer + 1
            for i in range(1, qubits, 2):
                if i + 1 < qubits:
                    gates.append(("CZ", (qubitList[i], qubitList[i + 1])))
            for i in range(1, qubits - 1):
                gates.append(("Ry", (parameters[layer][i - 1], qubitList[i])))
            layer = layer + 1
    return gates


def getControlledFixedAnsatzGates(qubits: int, parameters: List[List[float]]) -> tuple[str, tuple[float, float]]:
    """
    Generates a list of controlled gates for the controlled fixed ansatz.

    Parameters:
    - qubits (int): Number of qubits in the circuit.
    - parameters (List[List[float]]): List of parameters for the ansatz gates.

    Returns:
    - List of controlled gates as tuples, where each tuple is (gate_type, parameters).
    """
    gates = getFixedAnsatzGates(qubits, parameters)
    controlledGates = []
    auxiliaryQubit = 0
    # Makes controlled fixed ansatz gate using simple rules mentioned in:
    # Bakalaurinis_darbas.pdf page 14 Section: "Specialusis Hadamardo testas"
    for i in range(len(gates)):
        if gates[i][0] == "Ry":
            controlledGates.append(("CRy", (gates[i][1][0], (0, gates[i][1][1] + 1))))
        elif gates[i][0] == "Rx":
            controlledGates.append(("CRx", (gates[i][1][0], (0, gates[i][1][1] + 1))))
        elif gates[i][0] == "Rz":
            controlledGates.append(("CRz", (gates[i][1][0], (0, gates[i][1][1] + 1))))
        elif gates[i][0] == "CZ":
            controlledGates.append(("CCZ", (auxiliaryQubit, gates[i][1][0] + 1, gates[i][1][1] + 1)))
    return controlledGates
