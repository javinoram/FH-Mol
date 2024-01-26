from base import *
import pandas as pd

lr = float(sys.argv[1])
T = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
U = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
N = [2,3,4,5]

for n in N:
    for t in T:
        for u in U:
            print(u)
            p1, p2, p3, p4 = params(n, t, u, True, lr, 0)
            energy, theta = complete_flow_UCCSD(p1,p2,p3,p4)

            aux = pd.DataFrame(energy)
            aux.to_csv("FH/datos/adagrad"+str(lr)+"-EnergyClose"+str(n)+"t"+str(t)+"u"+str(u)+".csv")

            aux = pd.DataFrame(theta)
            aux.to_csv("FH/datos/adagrad"+str(lr)+"-ThetaClose"+str(n)+"t"+str(t)+"u"+str(u)+".csv")

            aux = [get_state_UCCSD(angle, p3) for angle in theta]
            aux = pd.DataFrame(aux)
            aux.to_csv("FH/datos/adagrad"+str(lr)+"-StateClose"+str(n)+"t"+str(t)+"u"+str(u)+".csv")