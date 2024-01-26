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
Funcion para ejecutar el VQE para la combinación Fermi-Hubbard y ansatz UCCSD.
input:
    p1: Diccionario de parametros del modelo Fermi-Hubbard.
    p2: Diccionario de parametros para la lattice del modelo Fermi-Hubbard.
    p3: Diccionario de los parametros del ansatz.
    p4: Diccionario de los parametros del optimizador.
output:
    energy: Lista con el valor de energia en cada iteracion del VQE (el ultimo elemento es el valor final entregado por el VQE).
    optimum: Lista con los valores de los parametros (los angulos) del ansatz en cada iteracion.
"""
def complete_flow_UCCSD(p1,p2,p3,p4):
    system = qs.vqe_fermihubbard(p1, p2)
    ansazt = qs.uccds_ansatz()
    ansazt.set_device( p3 )
    ansazt.set_node( p3 )
    ansazt.set_exitations( p3["electrons"], sz=0 )
    ansazt.set_state( p3["electrons"], sz=0 )
    system.set_node( ansazt.node, p3["interface"] )
    optimizer = qs.gradiend_optimizer(p4)
    optimizer.get_energy = True
    optimizer.get_params = True
    energy, optimum = optimizer.VQE(system.cost_function)
    return energy, optimum


"""
Funcion para obtener el vector de estado dentro del circuito.
input:
    angle: Lista de parametros del circuito
    ansatz_params: Parametros del circuito usados en el VQE
output:
    estado: Vector de numpy del estado dentro del circuito
"""
def get_state_UCCSD(angle, ansatz_params):
    ansatz_params["diff_method"] = "best"

    ansazt = qs.uccds_ansatz()
    ansazt.set_device( ansatz_params )
    ansazt.set_node( ansatz_params )
    ansazt.set_exitations( ansatz_params["electrons"], sz=0 )
    ansazt.set_state( ansatz_params["electrons"], sz=0 )
    return np.round( np.real( ansazt.get_state( angle ) ), 7)


"""
Funcion para ejecutar el VQE para la combinación Fermi-Hubbard y ansatz kUpCCGSD.
input:
    p1: Diccionario de parametros del modelo Fermi-Hubbard.
    p2: Diccionario de parametros para la lattice del modelo Fermi-Hubbard.
    p3: Diccionario de los parametros del ansatz.
    p4: Diccionario de los parametros del optimizador.
output:
    energy: Lista con el valor de energia en cada iteracion del VQE (el ultimo elemento es el valor final entregado por el VQE).
    optimum: Lista con los valores de los parametros (los angulos) del ansatz en cada iteracion.
"""
def complete_flow_KUPCCGSD(p1,p2,p3,p4):
    system = qs.vqe_fermihubbard(p1, p2)

    ansazt = qs.kupccgsd_ansatz()
    ansazt.set_device( p3 )
    ansazt.set_node( p3 )
    ansazt.set_state( p3["electrons"], sz=0 )
    system.set_node( ansazt.node, p3["interface"] )
    optimizer = qs.gradiend_optimizer(p4)
    optimizer.get_energy = True
    optimizer.get_params = True
    energy, optimum = optimizer.VQE(system.cost_function)
    return energy, optimum


"""
Funcion para obtener el vector de estado dentro del circuito.
input:
    angle: Lista de parametros del circuito
    ansatz_params: Parametros del circuito usados en el VQE
output:
    estado: Vector de numpy del estado dentro del circuito
"""
def get_state_kUpCCGSD(angle, ansatz_params):
    ansatz_params["diff_method"] = "best"

    ansazt = qs.kupccgsd_ansatz()
    ansazt.set_device( ansatz_params )
    ansazt.set_node( ansatz_params )
    ansazt.set_state( ansatz_params["electrons"], sz=0 )
    return np.round( np.real( ansazt.get_state( angle ) ), 7)


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
            "theta":["adagrad", lr]
        }
    else:
        a,b = qml.kUpCCGSD.shape(k=ansatz_params["repetitions"], n_wires=params["sites"]*2, delta_sz=0)
        minimizate_params = {
            "maxiter": 1000,
            "tol": 1e-6,
            "number":  a*b,
            "theta":["adagrad", lr]
        }
    return params, params_lat, ansatz_params, minimizate_params
    

