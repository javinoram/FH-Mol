from base import *
import pandas as pd

#Lectura del parametro de learning rate indicado en la ejecucion del archivo
lr = float(sys.argv[1])

#Vector de las diferentes distancias entre moleculas
distance = np.linspace(1.0, 8.0, 21)

for d in distance:
        #Construccion de los diccionarios de parametros
        symbols, coor, p1, p2, p3 = params3H(d, 'sto-3g', lr, 0)

        #Ejecucion del VQE bajo las condiciones dadas por los diccionarios
        energy, theta = complete_flow_UCCSD(symbols, coor, p1, p2, p3)
        
        #Almacenamientos de los resultados del VQE en archivos .csv (energia, parametros del ansatz y el estado
        #del circuito en cada iteracion)
        aux = pd.DataFrame(energy)
        aux.to_csv("Mol/datos/adagrad"+str(lr)+"-3HEnergy"+str(d)+".csv")
        aux = pd.DataFrame(theta)
        aux.to_csv("Mol/datos/adagrad"+str(lr)+"-3HTheta"+str(d)+".csv")
        aux = [get_state_UCCSD(angle, p2) for angle in theta]
        aux = pd.DataFrame(aux)
        aux.to_csv("Mol/datos/adagrad"+str(lr)+"-3HStates"+str(d)+".csv")