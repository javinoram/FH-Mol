import vqesimulation as qs
import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import jax


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

def get_state_UCCSD(angle, ansatz_params):
    ansatz_params["diff_method"] = "best"

    ansazt = qs.uccds_ansatz()
    ansazt.set_device( ansatz_params )
    ansazt.set_node( ansatz_params )
    ansazt.set_exitations( ansatz_params["electrons"], sz=0 )
    ansazt.set_state( ansatz_params["electrons"], sz=0 )
    return np.round( np.real( ansazt.get_state( angle ) ), 7)


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

def get_state_kUpCCGSD(angle, ansatz_params):
    ansatz_params["diff_method"] = "best"

    ansazt = qs.kupccgsd_ansatz()
    ansazt.set_device( ansatz_params )
    ansazt.set_node( ansatz_params )
    ansazt.set_state( ansatz_params["electrons"], sz=0 )
    return np.round( np.real( ansazt.get_state( angle ) ), 7)

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
        singles, doubles = qml.qchem.excitations(params["sites"]*2, params["sites"], 0)
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
    

