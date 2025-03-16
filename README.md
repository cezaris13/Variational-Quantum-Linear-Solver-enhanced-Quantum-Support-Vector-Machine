# Bachelor thesis on Variational quantum system of linear equations applied for support vector machines algorithm

## Main contents of this repository:
- SVMLearning
  Jupyter notebook to learn the Support vector machines algotithm
- VQLS-QSVM-Implementation
  Implementation of https://arxiv.org/pdf/2309.07770 article. Added generalization for the $n$ qubits and some solution vector post processing.
- Bakalaurinis_darbas.pdf
    - Contains the bachelor thesis(in lithuanian) where the VQLS-SVM algorithm is explored and experiment results can be seen.

#### VQLS-SVM algorithm
1. Take training data and prepare it to be [LS-SVM](https://en.wikipedia.org/wiki/Least-squares_support_vector_machine#:~:text=Elimination%20of,quadratic%20programming%20problem%3A) matrix form
1. Convert LS-SVM matrix into all combinations of

    $$|\langle b | \Phi \rangle|^2 \ = \ \displaystyle\sum_{m} \displaystyle\sum_{n} c_m c_n \langle 0 | U^{\dagger} A_n V(k) | 0 \rangle \langle 0 | U^{\dagger} A_m V(k) | 0 \rangle$$

    and

    $$\langle \Phi | \Phi \rangle \ = \ \displaystyle\sum_{m} \displaystyle\sum_{n} c_m^{*} c_n \langle 0 | V(k)^{\dagger} A_m^{\dagger} A_n V(k) |0\rangle$$

1. Calculate cost function using

    $$\hat{C}_P \ =  1 \ - \ \frac{|\langle b | \Phi \rangle|^2}{\langle \Phi | \Phi \rangle}$$
1. Apply minimization function (COBYLA, etc.) to modify $\alpha$ values, for the ansatz gate.

1. When $\alpha_{min}$ is retrieved apply once more to ansatz circuit to retrieve normalized estimated solution vector.

1. Apply post processing to the normalized estimated solution vector to retrieve estimated vector.

More details of how the VQLS algorithm works can be found [here](/VQLS-QSVM-Implementation/CodeExamples/VQLS.ipynb).

### SimulationTests folder

All of these test results have been used in the bachelor thesis.

- [TPDTest](/VQLS-QSVM-Implementation/SimulationTests/TPDTest.ipynb)
    Used to compare [Sparse Pauli operator](https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.SparsePauliOp) and [Tensorized Pauli decomposition (TPD)](https://arxiv.org/pdf/2310.13421) performance using qubits up to 10.
- [VQLSOptimizersTest](./VQLS-QSVM-Implementation/SimulationTests/VQLSOptimizersTest.ipynb)
    Ran tests for qubit size up to 5 qubits with different non gradient and gradient optimizers (COBYLA,SLSQP, BFGS, L-BFGS-B,trust-constr)
- [VQLSSVMTest](/VQLS-QSVM-Implementation/SimulationTests/VQLSSVMTest.ipynb)
    Testing VQLS-SVM algorithm with qubits up to 5, with iris and breast cancer datasets.
- [ThreadsVQLS](/VQLS-QSVM-Implementation/SimulationTests/ThreadsVQLS.ipynb)
    Notebook used to test verious job size and thead combinations to find which option is the fastest

## Update

Removed ancilla qubit for special Hadamard test, replaced cz gate encoding with ccz.

## Note

For some reason, after qiskit= 1.1.2 the minimization function goes below 0.