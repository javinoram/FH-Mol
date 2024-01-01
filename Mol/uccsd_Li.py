from base import *
import pandas as pd

distance = np.linspace(1.0, 8.0, 21)

for d in distance:
        print(d)
        symbols, coor, p1, p2, p3 = paramsLi(d, 'sto-3g', 0.2, 0)
        energy, theta = complete_flow_UCCSD(symbols, coor, p1, p2, p3)

        aux = pd.DataFrame(energy)
        aux.to_csv("datos/Mol/LiEnergy"+str(d)+".csv")

        aux = pd.DataFrame(theta)
        aux.to_csv("datos/Mol/LiTheta"+str(d)+".csv")

        aux = [get_state_UCCSD(angle, p2) for angle in theta]
        aux = pd.DataFrame(aux)
        aux.to_csv("datos/Mol/LiStates"+str(d)+".csv")