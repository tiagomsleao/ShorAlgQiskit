""" This file allows to test the various QFT implemented. The user must specify:
    1) The number of qubits it wants the QFT to be implemented on
    2) The kind of QFT want to implement, among the options:
        -> Normal QFT with SWAP gates at the end
        -> Normal QFT without SWAP gates at the end
        -> Inverse QFT with SWAP gates at the end
        -> Inverse QFT without SWAP gates at the end
    The user must can also specify, in the main function, the input quantum state. By default is a maximal superposition state
    This file uses as simulator the local simulator 'statevector_simulator' because this simulator saves
    the quantum state at the end of the circuit, which is exactly the goal of the test file. This simulator supports sufficient 
    qubits to the size of the QFTs that are going to be used in Shor's Algorithm because the IBM simulator only supports up to 32 qubits
"""

""" Imports from qiskit"""
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, IBMQ, BasicAer

import sys

""" Imports to Python functions """
import math
import array
import fractions
import numpy as np

""" Function to create QFT """
def create_QFT(circuit,up_reg,n,with_swaps):
    i=n-1
    """ Apply the H gates and Cphases"""
    """ The Cphases with |angle| < threshold are not created because they do 
    nothing. The threshold is put as being 0 so all CPhases are created,
    but the clause is there so if wanted just need to change the 0 of the
    if-clause to the desired value """
    while i>=0:
        circuit.h(up_reg[i])        
        j=i-1  
        while j>=0:
            if (np.pi)/(pow(2,(i-j))) > 0:
                circuit.cu1( (np.pi)/(pow(2,(i-j))) , up_reg[i] , up_reg[j] )
                j=j-1   
        i=i-1  

    """ If specified, apply the Swaps at the end """
    if with_swaps==1:
        i=0
        while i < ((n-1)/2):
            circuit.swap(up_reg[i], up_reg[n-1-i])
            i=i+1

""" Function to create inverse QFT """
def create_inverse_QFT(circuit,up_reg,n,with_swaps):
    """ If specified, apply the Swaps at the beggining"""
    if with_swaps==1:
        i=0
        while i < ((n-1)/2):
            circuit.swap(up_reg[i], up_reg[n-1-i])
            i=i+1
    
    """ Apply the H gates and Cphases"""
    """ The Cphases with |angle| < threshold are not created because they do 
    nothing. The threshold is put as being 0 so all CPhases are created,
    but the clause is there so if wanted just need to change the 0 of the
    if-clause to the desired value """
    i=0
    while i<n:
        circuit.h(up_reg[i])
        if i != n-1:
            j=i+1
            y=i
            while y>=0:
                 if (np.pi)/(pow(2,(j-y))) > 0:
                    circuit.cu1( - (np.pi)/(pow(2,(j-y))) , up_reg[j] , up_reg[y] )
                    y=y-1   
        i=i+1

""" Function to properly print the final state of the simulation """
""" This is only possible in this way because the program uses the statevector_simulator """
def show_good_coef(results, n):
    i=0
    max = pow(2,n)
    """ Iterate to all possible states """
    while i<max:
        binary = bin(i)[2:].zfill(n)
        number = results.item(i)
        number = round(number.real, 3) + round(number.imag, 3) * 1j
        """ Print the respective component of the state if it has a non-zero coeficient """
        if number!=0:
            print('|{}>'.format(binary),number)
        i=i+1

""" Main program """
if __name__ == '__main__':

    """ Select how many qubits want to apply the QFT on """
    n = int(input('\nPlease select how many qubits want to apply the QFT on: '))

    """ Select the kind of QFT to apply using the variable what_to_test:
        what_to_test = 0: Apply normal QFT with the SWAP gates at the end
        what_to_test = 1: Apply normal QFT without the SWAP gates at the end
        what_to_test = 2: Apply inverse QFT with the SWAP gates at the end
        what_to_test = 3: Apply inverse QFT without the SWAP gates at the end
    """
    print('\nSelect the kind of QFT to apply:')
    print('Select 0 to apply normal QFT with the SWAP gates at the end')
    print('Select 1 to apply normal QFT without the SWAP gates at the end')
    print('Select 2 to apply inverse QFT with the SWAP gates at the end')
    print('Select 3 to apply inverse QFT without the SWAP gates at the end\n')

    what_to_test = int(input('Select your option: '))

    if what_to_test<0 or what_to_test>3:
        print('Please select one of the options')
        exit()
    
    print('\nTotal number of qubits used: {0}\n'.format(n))

    print('Please check source file to change input quantum state. By default is a maximal superposition state with |+> in every qubit.\n')

    """ Create quantum and classical registers """
    quantum_reg = QuantumRegister(n)

    classic_reg = ClassicalRegister(n)

    """ Create Quantum Circuit """
    circuit = QuantumCircuit(quantum_reg, classic_reg)

    """ Create the input state desired
        Please change this as you like, by default we put H gates in every qubit, 
        initializing with a maximimal superposition state
    """
    #circuit.h(quantum_reg)

    """ Test the right QFT according to the variable specified before"""
    if what_to_test == 0:
        create_QFT(circuit,quantum_reg,n,1)
    elif what_to_test == 1:
        create_QFT(circuit,quantum_reg,n,0)
    elif what_to_test == 2:
        create_inverse_QFT(circuit,quantum_reg,n,1)
    elif what_to_test == 3:
        create_inverse_QFT(circuit,quantum_reg,n,0)
    else:
        print('Noting to implement, exiting program')
        exit()

    """ Simulate the created Quantum Circuit """
    simulation = execute(circuit, backend=BasicAer.get_backend('statevector_simulator'),shots=1)

    """ Get the results of the simulation in proper structure """
    sim_result=simulation.result()

    """ Get the statevector of the final quantum state """
    outputstate = sim_result.get_statevector(circuit, decimals=3)

    """ Print final quantum state to user """
    print('The final state after applying the QFT is:\n')
    show_good_coef(outputstate,n)