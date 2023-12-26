'''
Outdated example of VQE simulating a lithium hydride molecule.
It is pulling apart lithium hydride molecule and cacualting ground state energy at each distance.
The distance that gives us minimum energy is the actual interatomic distance
that exists in the lithium and hydrogen atoms.
Source: https://www.youtube.com/watch?v=Z-A6G0WVI9w

'''
from pickletools import optimize
import numpy as np              # basic numerics package
import pylab                    # plotting
import copy                     # convient
from qiskit import BasicAer     # quatum computer simulator
from qiskit.aqua import aqua_globals, QuantumInstance               # tools from aqua (able to run experiemnt)
from qiskit.aqua.algorithms import NumPyMinimunEigensolver, VQE     # exact energies based off of classical caculations (for comparing towards VQE) 
from qiskit.aqua.components.optimizers import SLSQP                 # classical optimizer to update anzats
from qiskit.chemistry.components.initial_states import HartreeFock  # inital anzats
from qiskit.chemistry.components.variational_forms import UCCSD     # varries hartreefock guess into VQE anzats
from qiskit.chemistry.drivers import pySCFDriver                    # sets up molecule
from qiskit.chemistry.cor import Hamiltonian, QubitMappingType      #  helps with mapping

molecule = 'H .0 .0 -{0}; Li .0 .0 {0}'     # modeling lithium hydride
distances =  np.arange(0.5, 4.25, 0.25)     # distance of 5 - 4 angstrums intervals of .25 (10 to the minus 10 meters)
vqe_energies = []                           # ground state energies calculated by VQE
hf_energies = []                            # init guess that hasnt been optimize by VQE
exact_energies = []                         # classical solver

for i, d in enumerate(distances):           # loop over variaces distances and compute VQE
    print('step', i)                        # debug which step we are on

    # set up experiment
    driver = pySCFDriver(molecule.formate(d/2), basis = 'sto3g')    # computes enrgy of each distance
    qmolecule = driver.run()                                        # quatum milecule
    operator = Hamiltonian(qubit_mapping = QubitMappingType.PARITY,
                           two_qubit_reduction = True, 
                           freeze_core = True,
                           orbital_reduction = [-3, 2])             # encode info of molecule into quantum comptuer
    qubit_op, aux_ops = operator.run(qmolecule)

    # extact classical result
    exact_result = NumPyMinimunEigensolver(qubit_op, aux_operators = aux_ops)
    exact_result = operator.process_algorithm_result(exact_result)

    # VQE
    optimizer = SLSQP(maxiter = 1000)
    initial_state = HartreeFock(operator.molecule_info['num_orbitals'],
                                operator.molecule_info['num_particles'],
                                qubit_mapping = operator._qubit_mapping,
                                two_qubit_reduction = operator._two_qubit_reduction)
    var_form = UCCSD(num_orbitals = operator.molecule_info['num_orbitals'],
                     num_particles = operator.molecule_info['num_particles'],
                     initial_state = initial_state,
                     qubit_mapping = operator._qubit_mapping,
                     two_qubit_reduction = operator._two_qubit_reduction)
    algo = VQE(qubit_op, var_form, optimizer, aux_operators = aux_ops)                      # VQE parematers ready to go                    
    vqe_result = algo.run(QuantumInstance(BasicAer.get_backend('statevector_simulator')))   # runs experiment
    vqe_result = operator.process_algorithm_result(vqe_result)                              # gets all energies

    exact_energies.append(exact_result.energy)
    vqe_energies.append(vqe_result.energy)
    hf_energies.append(vqe_result.hartree_fock_energy)

# make graphs
pylab.plot(distances, hf_energies, label = 'Hartree-Fock')
pylab.plot(distances, vqe_energies, 'o', label = 'VQE')
pylab.plot(distances, exact_energies, 'x', label = 'Exact')

pylab.xlabel('Iteratomic distance')
pylab.ylabel('Energy')
pylab.title('LiH Ground State Energy')
pylab.legend(loc = 'upper right')

