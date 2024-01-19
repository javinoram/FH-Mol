from base import *
import pandas as pd

T = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
U = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
N = [6]

for n in N:
    for t in T:
        for u in U:
            print(u)
            p1, p2, p3, p4 = params(n, t, u, False, 0.2, 0)
            system = qs.vqe_fermihubbard(p1, p2)
            energy, states = energies_and_states(system.hamiltonian, system.qubits)
            energy = np.round(energy, 7)
            states = np.round(np.real(states), 7)
            aux = pd.DataFrame(energy)
            aux.to_csv("datos/FMBarrido/ExactoEnergyOpen"+str(n)+"t"+str(t)+"u"+str(u)+".csv")

            aux = [states[:,i] for i in range(len(energy))]
            aux = pd.DataFrame(aux)
            aux.to_csv("datos/FMBarrido/ExactoStateOpen"+str(n)+"t"+str(t)+"u"+str(u)+".csv")