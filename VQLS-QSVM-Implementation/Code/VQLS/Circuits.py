from qiskit import QuantumCircuit, Aer, transpile
from qiskit.quantum_info import PauliList
from typing import List
import contextlib
import io

from Code.VQLS.Ansatz import fixedAnsatz, controlledFixedAnsatz
from Code.VQLS.LCU import convertMatrixIntoCircuit
from Code.VQLS.LabelVector import controlledLabelVectorCircuit


def getSolutionVector(circ: QuantumCircuit, qubits: int, outF: list):
    fixedAnsatz(circ, qubits, outF)
    circ.save_statevector() # this might be the problem

    backend = Aer.get_backend("aer_simulator")

    with contextlib.redirect_stdout(io.StringIO()):
        t_circ = transpile(circ, backend)
    job = backend.run(t_circ)

    result = job.result()
    return result.get_statevector(circ, decimals=10)


def hadamardTest(
    circ: QuantumCircuit,
    paulis: PauliList,
    qubits: int,
    parameters: List[List[float]],
):
    auxiliaryIndex = 0
    circ.h(auxiliaryIndex)

    circ.barrier()

    fixedAnsatz(circ, qubits, parameters, offset=1)

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
    qubits: int,
    parameters: List[List[float]],
    weights: List[float],
):
    auxiliaryIndex = 0
    circ.h(auxiliaryIndex)

    circ.barrier()

    controlledFixedAnsatz(circ, qubits, parameters)

    circ.barrier()

    convertMatrixIntoCircuit(
        circ,
        paulis,
        controlled=True,
        auxiliaryQubit=auxiliaryIndex,
        showBarriers=False,
    )

    circ.barrier()

    controlledLabelVectorCircuit(circ, auxiliaryIndex, qubits, weights)

    circ.barrier()

    circ.h(auxiliaryIndex)
