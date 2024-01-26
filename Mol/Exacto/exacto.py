from base import *
import pandas as pd


distance = np.linspace(1.0, 8.0, 21)

for d in distance:
        print(d)
        symbols, coor, p1, p2, p3 = params3H(d, 'sto-3g', 0.2, 0)
        system = qs.vqe_molecular(symbols, coor, p1)

        energy, states = energies_and_states(system.hamiltonian, system.qubits)
        energy = np.round(energy, 7)
        states = np.round(np.real(states), 7)

        aux = pd.DataFrame(energy)
        aux.to_csv("Mol/datos/exacto-3HEnergy"+str(d)+".csv")

        aux = [states[:,i] for i in range(len(energy))]
        aux = pd.DataFrame(aux)
        aux.to_csv("Mol/datos/exacto-3HState"+str(d)+".csv")



distance = np.linspace(1.0, 8.0, 21)
for d in distance:
        print(d)
        symbols, coor, p1, p2, p3 = paramsLi(d, 'sto-3g', 0.2, 0)
        system = qs.vqe_molecular(symbols, coor, p1)

        energy, states = energies_and_states(system.hamiltonian, system.qubits)
        energy = np.round(energy, 7)
        states = np.round(np.real(states), 7)

        aux = pd.DataFrame(energy)
        aux.to_csv("Mol/datos/exacto-LiHEnergy"+str(d)+".csv")

        aux = [states[:,i] for i in range(len(energy))]
        aux = pd.DataFrame(aux)
        aux.to_csv("Mol/datos/exacto-LiHState"+str(d)+".csv")