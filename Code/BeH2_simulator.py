import qiskit
import numpy as np
import qiskit_nature
from qiskit import Aer
import matplotlib.pyplot as plt
import qiskit_nature.algorithms
from qiskit.algorithms import VQE
from ibm_quantum_widgets import *
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit_aer import AerSimulator
from warnings import filterwarnings
from qiskit_nature.drivers import Molecule
from qiskit_nature.settings import settings
import qiskit_nature.drivers.second_quantization
import qiskit_nature.problems.second_quantization
import qiskit_nature.transformers.second_quantization.electronic

filterwarnings('ignore')
settings.dict_aux_operators = False

#====================================================================#
# Prepares Qubit Operator For Quatum Simulations of Molecular System #
#====================================================================#
def get_qubit_op(molecule, remove_orbitals): 

    # translates molecular info into compatible formate
    driver = qiskit_nature.drivers.second_quantization.ElectronicStructureMoleculeDriver(
        molecule = molecule,
        basis = "sto3g",
        driver_type = qiskit_nature.drivers.second_quantization.ElectronicStructureDriverType.PYSCF)

    # encapsulate electronic structure problem into object
    problem = qiskit_nature.problems.second_quantization.ElectronicStructureProblem(
        driver,
        remove_orbitals)

    second_q_ops = problem.second_q_ops()           # gets 2nd quantized operators (fermonic creation & annihilation operators)
    num_spin_orbitals = problem.num_spin_orbitals   # gets number of spin orbitals (cosider both up & down electrons)
    num_particles = problem.num_particles           # total number of electrons in system
    hamiltonian = second_q_ops[0]                   # gets hamiltonian operator (contains systems totlal energy)

    # maps fermionic operators to qubit operators (using pauli operators X,Y,Z)
    mapper = qiskit_nature.mappers.second_quantization.ParityMapper()  

    # applies compatible operators and two qubit reduction to the qubit operator
    converter = qiskit_nature.converters.second_quantization.QubitConverter(mapper, two_qubit_reduction = True)
    reducer = qiskit.opflow.TwoQubitReduction(num_particles)
    qubit_op = converter.convert(hamiltonian)
    qubit_op = reducer.convert(qubit_op)

    return qubit_op, num_particles, num_spin_orbitals, problem, converter

#==================================================================#
# Finds The Exact Energy Level of The Current Interatomic Distance #
#==================================================================#
def exact_solver(problem, converter):
    solver = qiskit_nature.algorithms.NumPyMinimumEigensolverFactory()          # instantiates a classical eigensolver
    calc = qiskit_nature.algorithms.GroundStateEigensolver(converter, solver)   # instance of eigensolver
    result = calc.solve(problem)                                                # object that holds info (energy, wavefunction, etc)
    return result

#=================================================#
# Graphs The Energy Levels Found At Each Distance #
#=================================================#
def print_results(distances, exact_energies, vqe_energies):
    plt.title("Grond State Energy Levels of Beryllium Hydride (BeH2)")
    plt.plot(distances, vqe_energies, 'x', label = "VQE Energy")
    plt.plot(distances, exact_energies, label = "Exact Energy")
    plt.xlabel("Atomic Distance (Angstrom)")
    plt.ylabel("Energy")
    plt.legend()
    plt.show()

def main():
    vqe_energies = []                                               # holds vqe energies at each distance
    exact_energies = []                                             # holds exact energies at each distance
    distances = np.arange(0.5, 4.25, 0.25)                          # from 0.5 t0 4.0 with step 0.2 angstroms
    optimizer = qiskit.algorithms.optimizers.SLSQP(maxiter = 5)     # SLSQP optimizer with max 5 iterations
    backend = qiskit.BasicAer.get_backend("statevector_simulator")  # quatum simulator machine

    #  Iterates Through Each Interatomic Distance  #
    for dist in distances:
        molecule = Molecule(                
            geometry = [
                ["Be", [0.0, 0.0, 0.0] ],
                ["H", [dist, 0.0, 0.0] ],
                ["H", [dist, dist, 0.0] ]
            ],
            multiplicity = 1,  # = 2*spin+1
            charge = 0,
        )

        (qubit_op, num_particles, num_spin_orbitals, problem, converter) = get_qubit_op(molecule,
            [qiskit_nature.transformers.second_quantization.electronic.FreezeCoreTransformer(
            freeze_core=True, remove_orbitals=[-3,-2])])

        result = exact_solver(problem, converter)
        exact_energies.append(result.total_energies[0].real)
        init_state = qiskit_nature.circuit.library.HartreeFock(num_spin_orbitals, num_particles, converter)
        var_form = qiskit_nature.circuit.library.UCCSD(converter,
                        num_particles,
                        num_spin_orbitals,
                        initial_state = init_state)

        vqe = VQE(var_form, optimizer, quantum_instance = backend)
        vqe_calc = vqe.compute_minimum_eigenvalue(qubit_op)
        vqe_result = problem.interpret(vqe_calc).total_energies[0].real
        vqe_energies.append(vqe_result)

        print(f"Interatomic Distance: {np.round(dist, 2)}",
              f"VQE Result: {vqe_result:.5f}",
              f"Exact Energy: {exact_energies[-1]:.5f}")

    print("All energies have been calculated")
    print_results(distances, exact_energies, vqe_energies)
    
if __name__ == "__main__":
    main()