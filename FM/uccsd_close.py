from base import *
import pandas as pd

t = 1
U = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
N = [3,4,5]

for n in N:
    for u in U:
        print(u)
        p1, p2, p3, p4 = params(n, t, u, True, 0.2, 0)
        energy, theta = complete_flow_UCCSD(p1,p2,p3,p4)

        aux = pd.DataFrame(energy)
        aux.to_csv("datos/FM/EnergyClose"+str(n)+"u"+str(u)+".csv")

        aux = pd.DataFrame(theta)
        aux.to_csv("datos/FM/ThetaClose"+str(n)+"u"+str(u)+".csv")

        aux = [get_state_UCCSD(angle, p3) for angle in theta]
        aux = pd.DataFrame(aux)
        aux.to_csv("datos/FM/StateClose"+str(n)+"u"+str(u)+".csv")