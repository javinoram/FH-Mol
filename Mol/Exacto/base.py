import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import sys

#Import de la libreria VqePy de forma local, no se instalo con pip
sys.path.append("../vqesimulation")
import quantumsim as qs


"""
Funcion para calcular los valores y vectores propios.
input:
    H: Hamiltoniano en formato de la libreria Pennylane.
    qubits: Entero que representa el numero de qubits usado para modelar el hamiltoniano. 
output:
    ee: arreglo numpy con los valores propios ordenados de menor a mayor.
    vv: matriz con los vectores propios del hamiltoniano ( para acceder a ellos se usa vv[:,i] ).
"""
def energies_and_states(H, qubits):
    H = np.array( qml.matrix(H, wire_order=[i for i in range(qubits)]) )
    ee, vv = np.linalg.eigh(H)
    return ee,vv


"""
Funcion para obtener el vector de posiciones de la molecula LiH
input:
    x: Distancia entre moleculas (bonding distance)
output:
    estado: Vector de numpy con las posiciones de los elementos
"""
def spaceLi(x):
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, x], requires_grad=True)


"""
Funcion para obtener el vector de posiciones de la molecula 3H+
input:
    x: Distancia entre moleculas (bonding distance)
output:
    estado: Vector de numpy con las posiciones de los elementos
"""
def space3H(x):
    return np.array([0.0, 0.0, 0.0, 0.0, x, 0.0, 0.0, x/2.0, x*np.sqrt(3.0/4.0)], requires_grad=True)


"""
Funcion para construir los parametros para ejecutar el VQE en la molecula LiH.
input:
    d: Distancia entre elementos
    basis: Basis set para modelar la molecula
    lr: Parametro learning rate para el optimizador
    flag: Valor entero para seleccionar el ansatz, 0 para el UCCSD y 1 para el kUpCCGSD.
output:
    symbols: Lista de simbolos de la molecula
    coordenadas: Vector de parametros de la posiciones de los elementos
    params: Parametros de la clase de molecula.
    ansatz_params: Parametros para la clase del ansatz.
    minimizate_params: Parametros para la clase del optimizador.
"""
def paramsLi(d, basis, lr, flag):
    symbols = ["Li", "H"]
    params = {
        'mapping': "jordan_wigner",
        'charge': 0, 
        'mult': 1,
        'basis': basis,
        'method': 'dhf',
        'active_electrons': 2,
        'active_orbitals': 5,
    }

    ansatz_params = {
        "repetitions": 1,
        "base": "lightning.qubit",
        "interface": "autograd",
        "electrons": params["active_electrons"],
        "qubits": params["active_orbitals"]*2,
        "diff_method": "adjoint",
        }
    
    if flag == 0:
        singles, doubles = qml.qchem.excitations(params["active_electrons"], params["active_orbitals"]*2, 0)
        singles = len(singles)
        doubles = len(doubles)
        minimizate_params = {
            "maxiter": 1000,
            "tol": 1e-6,
            "number":  singles + doubles,
            "theta":["generic", lr]
        }
    else:
        a,b = qml.kUpCCGSD.shape(k=ansatz_params["repetitions"], n_wires=params["active_orbitals"]*2, delta_sz=0)
        minimizate_params = {
            "maxiter": 1000,
            "tol": 1e-6,
            "number":  a*b,
            "theta":["generic", lr]
        }
    return symbols, spaceLi(d), params, ansatz_params, minimizate_params
    

"""
Funcion para construir los parametros para ejecutar el VQE en la molecula 3H+.
input:
    d: Distancia entre elementos
    basis: Basis set para modelar la molecula
    lr: Parametro learning rate para el optimizador
    flag: Valor entero para seleccionar el ansatz, 0 para el UCCSD y 1 para el kUpCCGSD.
output:
    symbols: Lista de simbolos de la molecula
    coordenadas: Vector de parametros de la posiciones de los elementos
    params: Parametros de la clase de molecula.
    ansatz_params: Parametros para la clase del ansatz.
    minimizate_params: Parametros para la clase del optimizador.
"""
def params3H(d, basis, lr, flag):
    symbols = ["H", "H", "H"]
    params = {
        'mapping': "jordan_wigner",
        'charge': 1, 
        'mult': 1,
        'basis': basis,
        'method': 'dhf',
    }

    ansatz_params = {
        "repetitions": 1,
        "base": "lightning.qubit",
        "interface": "autograd",
        "electrons": 2,
        "qubits": 6,
        "diff_method": "adjoint",
        }
    
    if flag == 0:
        singles, doubles = qml.qchem.excitations(2, 6, 0)
        singles = len(singles)
        doubles = len(doubles)
        minimizate_params = {
            "maxiter": 1000,
            "tol": 1e-6,
            "number":  singles + doubles,
            "theta":["generic", lr]
        }
    else:
        a,b = qml.kUpCCGSD.shape(k=ansatz_params["repetitions"], n_wires=6, delta_sz=0)
        minimizate_params = {
            "maxiter": 1000,
            "tol": 1e-6,
            "number":  a*b,
            "theta":["generic", lr]
        }
    return symbols, space3H(d), params, ansatz_params, minimizate_params