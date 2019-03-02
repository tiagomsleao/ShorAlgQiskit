""" This file allows to test the Multiplication blocks Ua. This blocks, when put together as explain in
the report, do the exponentiation. 
The user can change N, n, a and the input state, to create the circuit:
    
 up_reg        |+> ---------------------|----------------------- |+>
                                        |
                                        |
                                        |
                                 -------|---------
                    ------------ |               | ------------
 down_reg      |x>  ------------ |     Mult      | ------------  |(x*a) mod N>
                    ------------ |               | ------------
                                 -----------------       

Where |x> has n qubits and is the input state, the user can change it to whatever he wants
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

def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

def modinv(a, m):
    g, x, y = egcd(a, m)
    if g != 1:
        raise Exception('modular inverse does not exist')
    else:
        return x % m

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

"""Function that calculates the array of angles to be used in the addition in Fourier Space"""
def getAngles(a,N):
    s=bin(int(a))[2:].zfill(N) 
    angles=np.zeros([N])
    for i in range(0, N):
        for j in range(i,N):
            if s[j]=='1':
                angles[N-i-1]+=math.pow(2, -(j-i))
        angles[N-i-1]*=np.pi
    return angles

"""Creation of a doubly controlled phase gate"""
def ccphase(circuit,angle,ctl1,ctl2,tgt):
    circuit.cu1(angle/2,ctl1,tgt)
    circuit.cx(ctl2,ctl1)
    circuit.cu1(-angle/2,ctl1,tgt)
    circuit.cx(ctl2,ctl1)
    circuit.cu1(angle/2,ctl2,tgt)

"""Creation of the circuit that performs addition by a in Fourier Space"""
"""Can also be used for subtraction by setting the parameter inv to a value different from 0"""
def phiADD(circuit,q,a,N,inv):
    angle=getAngles(a,N)
    for i in range(0,N):
        if inv==0:
            circuit.u1(angle[i],q[i])
        else:
            circuit.u1(-angle[i],q[i])

"""Single controlled version of the phiADD circuit"""
def cphiADD(circuit,q,ctl,a,n,inv):
    angle=getAngles(a,n)
    for i in range(0,n):
        if inv==0:
            circuit.cu1(angle[i],ctl,q[i])
        else:
            circuit.cu1(-angle[i],ctl,q[i])
        
"""Doubly controlled version of the phiADD circuit""" 
def ccphiADD(circuit,q,ctl1,ctl2,a,n,inv):
    angle=getAngles(a,n)
    for i in range(0,n):
        if inv==0:
            ccphase(circuit,angle[i],ctl1,ctl2,q[i])
        else:
            ccphase(circuit,-angle[i],ctl1,ctl2,q[i])

"""Circuit that implements doubly controlled modular addition by a"""        
def ccphiADDmodN(circuit, q, ctl1, ctl2, aux, a, N, n):
    ccphiADD(circuit, q, ctl1, ctl2, a, n, 0)
    phiADD(circuit, q, N, n, 1)
    create_inverse_QFT(circuit, q, n, 0)
    circuit.cx(q[n-1],aux)
    create_QFT(circuit,q,n,0)
    cphiADD(circuit, q, aux, N, n, 0)
    
    ccphiADD(circuit, q, ctl1, ctl2, a, n, 1)
    create_inverse_QFT(circuit, q, n, 0)
    circuit.x(q[n-1])
    circuit.cx(q[n-1],aux)
    circuit.x(q[n-1])
    create_QFT(circuit,q,n,0)
    ccphiADD(circuit, q, ctl1, ctl2, a, n, 0)

"""Circuit that implements the inverse of doubly controlled modular addition by a"""
def ccphiADDmodN_inv(circuit, q, ctl1, ctl2, aux, a, N, n):
    ccphiADD(circuit, q, ctl1, ctl2, a, n, 1)
    create_inverse_QFT(circuit, q, n, 0)
    circuit.x(q[n-1])
    circuit.cx(q[n-1],aux)
    circuit.x(q[n-1])
    create_QFT(circuit, q, n, 0)
    ccphiADD(circuit, q, ctl1, ctl2, a, n, 0)
    cphiADD(circuit, q, aux, N, n, 1)
    create_inverse_QFT(circuit, q, n, 0)
    circuit.cx(q[n-1], aux)
    create_QFT(circuit, q, n, 0)
    phiADD(circuit, q, N, n, 0)
    ccphiADD(circuit, q, ctl1, ctl2, a, n, 1)

"""Circuit that implements single controlled modular multiplication by a"""
def cMULTmodN(circuit, ctl, q, aux, a, N, n):
    create_QFT(circuit,aux,n+1,0)
    for i in range(0, n):
        ccphiADDmodN(circuit, aux, q[i], ctl, aux[n+1], int(math.pow(2,i)*a) % N, N, n+1)
    create_inverse_QFT(circuit, aux, n+1, 0)

    for i in range(0, n):
        circuit.cswap(ctl,q[i],aux[i])

    a_inv = modinv(a, N)
    create_QFT(circuit, aux, n+1, 0)
    i = n-1
    while i >= 0:
        ccphiADDmodN_inv(circuit, aux, q[i], ctl, aux[n+1], int(math.pow(2,i)*a_inv) % N, N, n+1)
        i -= 1
    create_inverse_QFT(circuit, aux, n+1, 0)

""" Function to properly get the final state, it prints it to user """
""" This is only possible in this way because the program uses the statevector_simulator """
def get_final(results, number_aux, number_up, number_down):
    i=0
    """ Get total number of qubits to go through all possibilities """
    total_number = number_aux + number_up + number_down
    max = pow(2,total_number)   
    print('|aux>|top_register>|bottom_register>\n')
    while i<max:
        binary = bin(i)[2:].zfill(total_number)
        number = results.item(i)
        number = round(number.real, 3) + round(number.imag, 3) * 1j
        """ If the respective state is not zero, then print it and store the state of the register where the result we are looking for is.
        This works because that state is the same for every case where number !=0  """
        if number!=0:
            print('|{0}>|{1}>|{2}>'.format(binary[0:number_aux],binary[number_aux:(number_aux+number_up)],binary[(number_aux+number_up):(total_number)]),number)
            if binary[number_aux:(number_aux+number_up)]=='1':
                store = binary[(number_aux+number_up):(total_number)]
        i=i+1

    print(' ')

    return int(store, 2)

""" Main program """
if __name__ == '__main__':

    """ Select number N to do modN"""
    N = int(input('Please insert integer number N: '))
    print(' ')

    """ Get n value used in QFT, to know how many qubits are used """
    n = math.ceil(math.log(N,2))

    """ Select the value for 'a' """
    a = int(input('Please insert integer number a: '))
    print(' ')

    """ Please make sure the a and N are coprime"""
    if math.gcd(a,N)!=1:
        print('Please make sure the a and N are coprime. Exiting program.')
        exit()

    print('Total number of qubits used: {0}\n'.format(2*n+3))

    print('Please check source file to change input quantum state. By default is |2>.\n')

    """ Create quantum and classical registers """
    aux = QuantumRegister(n+2)
    up_reg = QuantumRegister(1)
    down_reg = QuantumRegister(n)

    aux_classic = ClassicalRegister(n+2)
    up_classic = ClassicalRegister(1)
    down_classic = ClassicalRegister(n)

    """ Create Quantum Circuit """
    circuit = QuantumCircuit(down_reg , up_reg , aux, down_classic, up_classic, aux_classic)

    """ Initialize with |+> to also check if the control is working"""
    circuit.h(up_reg[0])

    """ Put the desired input state in the down quantum register. By default we put |2> """
    circuit.x(down_reg[1])
    
    """ Apply multiplication""" 
    cMULTmodN(circuit, up_reg[0], down_reg, aux, int(a), N, n)

    """ Simulate the created Quantum Circuit """
    simulation = execute(circuit, backend=BasicAer.get_backend('statevector_simulator'),shots=1)
    
    """ Get the results of the simulation in proper structure """
    sim_result=simulation.result()

    """ Get the statevector of the final quantum state """
    outputstate = sim_result.get_statevector(circuit, decimals=3)

    """ Show the final state after the multiplication """
    after_exp = get_final(outputstate, n+2, 1, n)

    print('When control=1, value after exponentiation is in bottom quantum register: |{0}>'.format(after_exp))