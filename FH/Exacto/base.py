import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import sys

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
Funcion para construir los parametros para ejecutar el VQE en el modelo Fermi-Hubbard.
input:
    n: Cantidad de sitios del sistema para una cadena 1D.
    t: Parametro de hopping (float).
    u: Parametro de on-site potencial (float).
    p: Condiciones de borde del sistema, True significa periodicidad y False condiciones abiertas.
    lr: float para indicar el learning rate que usara el optimizador.
    flag: Valor entero para seleccionar el ansatz, 0 para el UCCSD y 1 para el kUpCCGSD.
output:
    params: Parametros del modelo Fermi-Hubbard.
    params_lat: Parametros de la lattice del modelo.
    ansatz_params: Parametros para la clase del ansatz.
    minimizate_params: Parametros para la clase del optimizador.
"""
def params(n, t, u, p, lr, flag):
    params = {
        "sites": n, 
        "hopping": t,
        "U": u, 
        }

    params_lat = {
        "bound": p,
        "lattice": "chain",
        "size": (1,n)
        }

    ansatz_params = {
        "repetitions": 1,
        "base": "lightning.qubit",
        "interface": "autograd",
        "electrons": params["sites"],
        "qubits": params["sites"]*2,
        "diff_method": "adjoint",
        }
    if flag == 0:
        singles, doubles = qml.qchem.excitations(params["sites"], params["sites"]*2, 0)
        singles = len(singles)
        doubles = len(doubles)
        minimizate_params = {
            "maxiter": 1000,
            "tol": 1e-6,
            "number":  singles + doubles,
            "theta":["generic", lr]
        }
    else:
        a,b = qml.kUpCCGSD.shape(k=ansatz_params["repetitions"], n_wires=params["sites"]*2, delta_sz=0)
        minimizate_params = {
            "maxiter": 1000,
            "tol": 1e-6,
            "number":  a*b,
            "theta":["generic", lr]
        }
    return params, params_lat, ansatz_params, minimizate_params
    

