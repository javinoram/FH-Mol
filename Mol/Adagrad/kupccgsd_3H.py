from base import *
import pandas as pd

distance = np.linspace(1.0, 8.0, 21)
lr = float(sys.argv[1])

for d in distance:
        print(d)
        symbols, coor, p1, p2, p3 = params3H(d, 'sto-3g', lr, 1)
        energy, theta = complete_flow_KUPCCGSD(symbols, coor, p1, p2, p3)

        aux = pd.DataFrame(energy)
        aux.to_csv("Mol/datos/adagrad"+str(lr)+"-K3HEnergy"+str(d)+".csv")

        aux = pd.DataFrame(theta)
        aux.to_csv("Mol/datos/adagrad"+str(lr)+"-K3HTheta"+str(d)+".csv")

        aux = [get_state_kUpCCGSD(angle, p2) for angle in theta]
        aux = pd.DataFrame(aux)
        aux.to_csv("Mol/datos/adagrad"+str(lr)+"-K3HStates"+str(d)+".csv")