from qiskit import QuantumCircuit
from qiskit.quantum_info import PauliList
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
from typing import List
from qiskit.circuit import ParameterVector
from qiskit_algorithms.optimizers import ADAM, SPSA, GradientDescent

from Code.Utils import splitParameters, getTotalAnsatzParameters, TriangleMatrix, prepareBackend
from Code.VQLS.Circuits import prepareCircuits

costHistory = []


def minimization(
    paulis: PauliList,
    coefficientSet: List[float],
    qubits: int,
    bVector: list,
    quantumSimulation: bool = True,
    method: str = "COBYLA",
    shots: int = 100000,
    iterations: int = 200,
    verbose: bool = True,
    layers: int = 3,
    threads: int = 1,
    jobs: int = 1,
    options: dict = {},
    threadingForCircuits: bool = False,
) -> List[List[float]]:
    '''
    Minimizes a cost function using quantum circuits and a classical optimization algorithm.

    This function performs a hybrid quantum-classical optimization to minimize a cost function.
    It uses a quantum backend to simulate or run quantum circuits, evaluates the cost function
    for a set of parameters, and iteratively minimizes it using a classical optimization method.

    Parameters:
    - paulis : PauliList
        A list of Pauli operators to be applied to the quantum circuits.

    - coefficientSet : List[float]
        A list of coefficients that correspond to the Pauli operators in the cost function.

    - qubits : int
        The number of qubits to be used in the quantum circuits.

    - bVector : list
        The vector \( b \) used in the minimization, typically representing the target or
        problem state in the quantum system.

    - quantumSimulation : bool, optional (default: True)
        If `True`, the function will run a quantum simulation with number of shots on specific backend. If `False`, it will execute the
        circuits using a statevector simulation.

    - method : str, optional (default: "COBYLA")
        The classical optimization method to use. Default is "COBYLA", but can be any method
        supported by the underlying optimization library.

    - shots : int, optional (default: 100000)
        The number of shots (repetitions) to perform when running quantum circuits,
        applicable when `quantumSimulation` is `True`.

    - iterations : int, optional (default: 200)
        The number of iterations for the classical optimization routine.

    - verbose : bool, optional (default: True)
        If `True`, the function will print out detailed information such as time taken
        to prepare circuits and to perform the minimization.

    - layers : int, optional (default: 3)
        The number of layers to use in the quantum circuit ansatz.

    - threads : int, optional (default: 1)
        The number of threads to use for parallelizing the quantum circuit execution,
        if supported by the backend.

    - jobs : int, optional (default: 1)
        The number of parallel jobs to use when running the quantum backend.

    - options : dict, optional (default: {})
        Additional options for the classical optimizer. It is usually used for specific parameters for classical optimizers.

    - threadingForCircuits : bool, optional (default: False)
        If `True`, parallelizes the preparation of circuits across multiple threads.

    Returns:
    - List[List[float]]
        A list of optimized circuit parameters grouped by qubit.

    Details:
    1. **Circuit Preparation**: The function first prepares the necessary quantum circuits for
       Hadamard and special Hadamard gates based on the provided Pauli operators and coefficients.
    2. **Backend Setup**: A quantum backend is selected based on the provided threading and job settings.
    3. **Minimization Process**: The minimization process is carried out using the selected
       classical optimizer, iterating for a set number of steps to find the optimal circuit parameters.
    4. **Cost History**: The global `costHistory` is updated during the minimization process to track
       the progress of the cost function.

    Example Usage:
    ```python
    paulis = PauliList(["X", "Y", "Z"])
    coefficientSet = [0.5, -1.2, 0.8]
    qubits = 4
    bVector = [1, 0, 0, 1]

    optimizedParams = minimization(
        paulis,
        coefficientSet,
        qubits,
        bVector,
        method="COBYLA",
        iterations=300,
        verbose=True
    )
    ```

    This example runs the minimization using a set of Pauli operators, corresponding coefficients,
    and a 4-qubit quantum system. It uses the "COBYLA" optimizer, performs 300 iterations, and
    outputs detailed information due to `verbose=True`.
    '''
    global costHistory
    costHistory = []

    start = time.time()
    (
        transpiledHadamardCircuit,
        parametersHadamard,
        transpiledSpecialHadamardCircuits,
        parametersSpecialHadamard,
    ) = prepareCircuits(
        paulis,
        bVector,
        qubits,
        quantumSimulation,
        layers,
        threads=threads,
        jobs=jobs,
        threading=threadingForCircuits,
    )
    end = time.time()
    if verbose:
        print("Time to prepare circuits:", end - start)

    backend = prepareBackend(threads, jobs)

    arguments = [
        coefficientSet,
        transpiledHadamardCircuit,
        parametersHadamard,
        transpiledSpecialHadamardCircuits,
        parametersSpecialHadamard,
        quantumSimulation,
        shots,
        backend
    ]

    start = time.time()
    circuitParameters = runMinimization(method, arguments, options, iterations, qubits, layers)
    end = time.time()

    if verbose:
        print("Time to minimize:", end - start)

    return splitParameters(circuitParameters, qubits, alternating=qubits != 3)


def runMinimization(method:str, arguments: list, options: dict, iterations: int, qubits: int, layers: int):
    '''
    Runs the minimization process using various optimization methods to minimize a cost function.

    This function applies different classical optimization algorithms to minimize the cost function
    for a quantum-classical hybrid system. Depending on the selected method, it will initialize the
    optimization with random parameters, normalize them, and iteratively minimize the cost using
    the specified optimizer.

    Parameters:
    - method : str
        The optimization method to use. Possible values include:
        - "ADAM": Adaptive Moment Estimation (a gradient-based optimizer).
        - "SPSA": Simultaneous Perturbation Stochastic Approximation.
        - "GD": Gradient Descent.
        - Scipy ther non gradient optimizers.

    - arguments : list
        A list of arguments required for the cost function. Includes quantum circuits,
        coefficients, backend details, and other relevant parameters.

    - options : dict
        A dictionary containing options for the optimizer, such as learning rate or perturbation size.
        If specific options are not provided, default values will be used.

    - iterations : int
        The maximum number of iterations for the optimizer.

    - qubits : int
        The number of qubits in the quantum system.

    - layers : int
        The number of layers in the quantum circuit ansatz, which affects the total number of parameters.

    Returns:
    - List[float]
        The optimized parameters after the minimization process.

    Details:
    1. **Parameter Initialization**:
        - The function first calculates the total number of parameters required for the ansatz
          using `getTotalAnsatzParameters`.
        - It then initializes these parameters randomly and normalizes them to ensure that the
          parameter space is well-formed for optimization.

    2. **Available Optimizers**:
        - **ADAM**: A gradient-based optimizer with optional learning rate (`lr`).
        - **SPSA**: A stochastic optimizer that uses a simultaneous perturbation approximation
          with options for `learning_rate` and `perturbation`.
        - **GD (Gradient Descent)**: A simple gradient descent optimizer with configurable `learning_rate`.
        - And other non gradient optimizers.

    3. **Cost Function**:
        - The cost function is wrapped using `funcWrapper`, which takes the current parameters and
          computes the cost using `calculateCostFunction`.

    4. **Optimizer Execution**:
        - The function runs the selected optimizer for the specified number of iterations,
          aiming to minimize the cost function.
        - The optimization process returns the optimized parameters (`x`).

    Example Usage:
    ```python
    method = "ADAM"
    arguments = [coefficientSet, transpiledHadamardCircuit, parametersHadamard, transpiledSpecialHadamardCircuits, parametersSpecialHadamard, quantumSimulation, shots, backend]
    options = {"lr": 0.01}
    iterations = 300
    qubits = 4
    layers = 3

    optimizedParams = runMinimization(method, arguments, options, iterations, qubits, layers)
    ```

    In this example, the minimization is run using the "ADAM" optimizer with 300 iterations and a learning rate of 0.01.
    '''
    totalParamsNeeded = getTotalAnsatzParameters(qubits, layers)
    x: List[float] = [
        float(random.randint(0, 3000)) for _ in range(0, totalParamsNeeded)
    ]
    x = x / np.linalg.norm(x)

    methods = ["ADAM", "SPSA", "GD"]

    if method in methods:
        def calculateCostFunctionWrapper(params): return calculateCostFunction(
            params, arguments)

        if method == "ADAM":
            lr = options["lr"] if "lr" in options else 0.05
            optimizer = ADAM(maxiter=iterations, lr=lr)
        elif method == "SPSA":
            learning_rate = (
                options["learning_rate"] if "learning_rate" in options else 0.2
            )
            perturbation = options["perturbation"] if "perturbation" in options else 0.1
            optimizer = SPSA(
                maxiter=iterations,
                learning_rate=learning_rate,
                perturbation=perturbation,
            )
        elif method == "GD":
            learning_rate = (
                options["learning_rate"] if "learning_rate" in options else 0.2
            )
            optimizer = GradientDescent(
                maxiter=iterations, learning_rate=learning_rate)

        out = optimizer.minimize(calculateCostFunctionWrapper, x0=x)
        return out.x

    out = minimize(
        calculateCostFunction,
        x0=x,
        args=arguments,
        method=method,
        options={"maxiter": iterations},
    )

    return out["x"]


def calculateCostFunction(parameters: list, args: list) -> float:
    '''
    Computes the cost function value for a quantum-classical hybrid system.

    This function calculates the cost based on variational quantum circuits by iterating through
    the list of Pauli operators and using both Hadamard and special Hadamard circuits.

    Parameters:
    - parameters : list
        A list of variational parameters used to assign values to the parameterized quantum circuits.
        (In the literature it is denoted by \u03B1)

    - args : list
        A list of additional arguments required for the cost function, structured as:
        1. coefficientSet : list
            Coefficients associated with the Pauli operators in the cost function.
        2. transpiledHadamardCircuit : list
            Transpiled quantum circuits for the Hadamard basis.
        3. parametersHadamard : list
            Parameters associated with the Hadamard circuits.
        4. transpiledSpecialHadamardCircuits : list
            Transpiled quantum circuits for special Hadamard gates.
        5. parametersSpecialHadamard : list
            Parameters associated with the special Hadamard circuits.
        6. isQuantumSimulation : bool
            Boolean flag indicating whether quantum simulation is enabled (true) or classical simulation is used (false).
        7. shots : int
            Number of shots (measurements) to be used if running on quantum hardware or simulation.
        8. backend : object
            The backend to run the quantum circuits (simulator or quantum device).

    Returns:
    - float
        The cost value calculated based on the results of the quantum circuits and the provided coefficients.

    Details:
    1. **Cost History Tracking**:
        - The function prints the current iteration and the cost in real-time.
        - It also stores the calculated cost in a global `costHistory` variable to track the evolution of the cost during the optimization process.

    2. **Circuit Binding**:
        - The function binds the provided variational parameters to the Hadamard and special Hadamard circuits by assigning them using the `assign_parameters` method.

    3. **Running Quantum Experiments**:
        - The bound circuits are executed on the provided backend (which could be either a quantum simulator or hardware) using the `runExperiments` function.
        - The function retrieves both the results for the Hadamard circuits and the special Hadamard circuits.

    4. **Cost Calculation**:
        - The main part of the cost calculation is divided into two sums: `overallSum1` and `overallSum2`.
        - **OverallSum1**:
            - This sum iterates over a triangular matrix structure where the lower triangular part corresponds to conjugates, which are handled appropriately to avoid redundant calculations.
        - **OverallSum2**:
            - A similar sum is calculated based on the special Hadamard circuit results. It also iterates over the Pauli coefficients to compute the weighted sum.

    5. **Final Cost**:
        - The final cost is computed as:
          \[
          \text{totalCost} = 1 - \frac{\text{overallSum2.real}}{\text{overallSum1.real}}
          \]
        - This value is appended to `costHistory` for tracking and optimization purposes.
    '''
    cost = 0
    if len(costHistory) > 0:
        cost = costHistory[len(costHistory) - 1]
    print("Iteration:", len(costHistory) + 1, ", cost:", cost, end="\r")
    overallSum1: float = 0
    overallSum2: float = 0

    coefficientSet: List[float] = args[0]
    transpiledHadamardCircuit: List[QuantumCircuit] = args[1]
    parametersHadamard: ParameterVector = args[2]
    transpiledSpecialHadamardCircuits: List[QuantumCircuit] = args[3]
    parametersSpecialHadamard: ParameterVector = args[4]
    isQuantumSimulation: bool = args[5]
    shots: int = args[6]
    backend = args[7]

    bindedHadamardGates: List[QuantumCircuit] = list(map(lambda x: x.assign_parameters({ parametersHadamard: parameters}), transpiledHadamardCircuit))
    bindedSpecHadamardGates: List[QuantumCircuit] = list(map(lambda x: x.assign_parameters({ parametersSpecialHadamard: parameters}), transpiledSpecialHadamardCircuits))
    lenPaulis = len(bindedSpecHadamardGates)
    resultsHadamard, resultsSpecialHadamard = runExperiments(bindedHadamardGates, bindedSpecHadamardGates, isQuantumSimulation, shots, backend)
    bindedHadamardGatesTriangle = TriangleMatrix(lenPaulis, resultsHadamard)

    # we have triangular matrix:
    # X X X X X
    # . X X X X
    # . . X X X
    # . . . X X
    # . . . . X
    # lower triangular matrix is calculated using this formula: <0|V(a)^d A_n^d A_m V(a)|0> = (<0|V(a)^d A_m^d A_n V(a)|0>) conjugate
    # c_n conj c_m <0|V(a)^d A_n^d A_m V(a)|0> = ( c_n conj c_m <0|V(a)^d A_m^d A_n V(a)|0>) conjugate
    for i in range(lenPaulis):
        for j in range(lenPaulis - i):
            outputstate = bindedHadamardGatesTriangle.getElement(i, i + j)
            m_sum = getMSum(isQuantumSimulation, outputstate, shots)
            multiply = coefficientSet[i] * coefficientSet[i + j]

            if j == 0:  # since the main diagional is not counted twice and  the list first element is the main diagional
                overallSum1 += multiply * (1 - (2 * m_sum))
            else:
                temp = multiply * (1 - (2 * m_sum))
                overallSum1 += np.conjugate(temp) + temp

    for i in range(lenPaulis):
        for j in range(lenPaulis - i):
            mult = 1
            indexArray = [i, j + i]
            for index in indexArray:
                outputstate = resultsSpecialHadamard[index]
                m_sum = getMSum(isQuantumSimulation, outputstate, shots)
                mult = mult * (1 - (2 * m_sum))
            multiply = coefficientSet[i] * coefficientSet[j + i]
            if j == 0:
                overallSum2 += multiply * mult
            else:
                tempSum = multiply * mult
                overallSum2 += tempSum + np.conjugate(tempSum)

    totalCost = 1 - float(overallSum2.real / overallSum1.real)
    costHistory.append(totalCost)
    return totalCost


def runExperiments(bindedHadamardGates: List[QuantumCircuit], bindedSpecHadamardGates: List[QuantumCircuit], isQuantumSimulation: bool, shots: int, backend):
    '''
    Run quantum experiments with given Hadamard gates and return the results.

    Parameters:
    - bindedHadamardGates (List[QuantumCircuit]): A list of Hadamard gate circuits to run.
    - bindedSpecHadamardGates (List[QuantumCircuit]): A list of special Hadamard gate circuits to run.
    - isQuantumSimulation (bool): Flag indicating if the backend is a quantum simulator.
    - shots (int): Number of shots or repetitions for each experiment.
    - backend: Backend to run the experiments on. Should have a `run` method that returns results.

    Returns:
    - tuple: A tuple containing two lists:
        - resultsHadamard (list): Results for the Hadamard gates, either counts or state vectors.
        - resultsSpecialHadamard (list): Results for the special Hadamard gates, either counts or state vectors.
    '''
    gates = bindedHadamardGates + bindedSpecHadamardGates

    if isQuantumSimulation:
        results = backend.run(gates, shots=shots).result()
        resultsHadamard = list(map(lambda x: results.get_counts(x), bindedHadamardGates))
        resultsSpecialHadamard = list(map(lambda x: results.get_counts(x), bindedSpecHadamardGates))
    else:
        results = backend.run(gates).result()
        resultsHadamard = list(map(lambda x: np.real(results.get_statevector(x, decimals=100)), bindedHadamardGates))
        resultsSpecialHadamard = list(map(lambda x: np.real(results.get_statevector(x, decimals=100)), bindedSpecHadamardGates))

    return resultsHadamard, resultsSpecialHadamard


def getMSum(isQuantumSimulation: bool, outputstate, shots: int) -> float:
    """
    Compute the M-sum from the output state of the experiments.

    Parameters:
    - isQuantumSimulation (bool): Flag indicating if the backend is a quantum simulator.
    - outputstate (dict or list): The output state from the experiment. Can be either a dictionary of counts
      (for quantum simulation) or a list of state vector probabilities (for non-quantum simulations).
    - shots (int): Number of shots or repetitions used in the experiment.

    Returns:
    - float: The computed M-sum.
    """
    if isQuantumSimulation:
        if "1" in outputstate.keys():
            m_sum = float(outputstate["1"]) / shots
        else:
            m_sum = 0
        return m_sum
    else:
        m_sum = 0
        for l in range(len(outputstate)):
            if l % 2 == 1:
                n = outputstate[l] ** 2
                m_sum += n
        return m_sum


def getApproximationValue(A: np.ndarray, b: np.array, xEstimated: np.array) -> float:
    """
    Calculate the approximation value of the given vectors and matrix.

    Parameters:
    - A (np.ndarray): A square matrix used in the approximation calculation.
    - b (np.array): A vector used in the approximation calculation.
    - xEstimated (np.array): The estimated vector used in the approximation calculation.

    Returns:
    - float: The approximation value as a float.
    """
    return ((b.dot(A.dot(xEstimated) / (np.linalg.norm(A.dot(xEstimated))))) ** 2).real


def getCostHistory():
    '''
    Retrieves the recorded cost history during optimization.

    Returns:
    costHistory : list
        A list or array containing the cost values at each optimization step.
    '''
    return costHistory


def plotCost():
    '''
    Plots the cost function over optimization steps.

    - The x-axis represents the number of optimization steps taken.
    - The y-axis represents the value of the cost function at each step.
    '''
    plt.style.use("seaborn-v0_8")
    plt.plot(costHistory, "g")
    plt.ylabel("Cost function")
    plt.xlabel("Optimization steps")
    plt.show()