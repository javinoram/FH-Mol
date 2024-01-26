from base import *
import pandas as pd

#Lectura del parametro de learning rate indicado en la ejecucion del archivo
lr = float(sys.argv[1])

#Vector de las diferentes distancias entre moleculas
distance = np.linspace(1.0, 8.0, 21)


for d in distance:
        #Construccion de los diccionarios de parametros
        symbols, coor, p1, p2, p3 = paramsLi(d, 'sto-3g', lr, 0)

        #Construccion del hamiltoniano
        system = qs.vqe_molecular(symbols, coor, p1)

        #Ejecucion del calculo de valores y vectores propios
        energy, states = energies_and_states(system.hamiltonian, system.qubits)
        energy = np.round(energy, 7)
        states = np.round(np.real(states), 7)

        #Almacenamiento de los valores y vectores propios en archivos .csv
        aux = pd.DataFrame(energy)
        aux.to_csv("Mol/datos/exacto-LiHEnergy"+str(d)+".csv")
        aux = [states[:,i] for i in range(len(energy))]
        aux = pd.DataFrame(aux)
        aux.to_csv("Mol/datos/exacto-LiHState"+str(d)+".csv")