import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import jax
import sys

sys.path.append("../vqesimulation")
import quantumsim as qs

def energies_and_states(H, qubits):
    H = np.array( qml.matrix(H, wire_order=[i for i in range(qubits)]) )
    ee, vv = np.linalg.eigh(H)
    return ee,vv

def complete_flow_UCCSD(symbols, coor, p1, p2, p3):
    system = qs.vqe_molecular(symbols, coor, p1)
    ansazt = qs.uccds_ansatz()
    ansazt.set_device( p2 )
    ansazt.set_node( p2 )
    ansazt.set_exitations( p2["electrons"], sz=0 )
    ansazt.set_state( p2["electrons"], sz=0 )
    system.set_node( ansazt.node, p2["interface"] )
    optimizer = qs.gradiend_optimizer(p3)
    optimizer.get_energy = True
    optimizer.get_params = True
    energy, optimum = optimizer.VQE(system.cost_function)
    return energy, optimum

def get_state_UCCSD(angle, ansatz_params):
    ansatz_params["diff_method"] = "best"

    ansazt = qs.uccds_ansatz()
    ansazt.set_device( ansatz_params )
    ansazt.set_node( ansatz_params )
    ansazt.set_exitations( ansatz_params["electrons"], sz=0 )
    ansazt.set_state( ansatz_params["electrons"], sz=0 )
    return np.round( np.real( ansazt.get_state( angle ) ), 7)


def complete_flow_KUPCCGSD(symbols, coor, p1, p2, p3):
    system = qs.vqe_molecular(symbols, coor, p1)
    ansazt = qs.kupccgsd_ansatz()
    ansazt.set_device( p2 )
    ansazt.set_node( p2 )
    ansazt.set_state( p2["electrons"], sz=0 )
    system.set_node( ansazt.node, p2["interface"] )
    optimizer = qs.gradiend_optimizer(p3)
    optimizer.get_energy = True
    optimizer.get_params = True
    energy, optimum = optimizer.VQE(system.cost_function)
    return energy, optimum

def get_state_kUpCCGSD(angle, ansatz_params):
    ansatz_params["diff_method"] = "best"

    ansazt = qs.kupccgsd_ansatz()
    ansazt.set_device( ansatz_params )
    ansazt.set_node( ansatz_params )
    ansazt.set_state( ansatz_params["electrons"], sz=0 )
    return np.round( np.real( ansazt.get_state( angle ) ), 7)




def spaceLi(x):
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, x], requires_grad=True)

def space3H(x):
    return np.array([0.0, 0.0, 0.0, 0.0, x, 0.0, 0.0, x/2.0, x*np.sqrt(3.0/4.0)], requires_grad=True)

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