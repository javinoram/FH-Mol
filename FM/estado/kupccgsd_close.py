from base import *
import pandas as pd

t = 1
U = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
N = [4]

for n in N:
    for u in U:
        print(u)
        p1, p2, p3, p4 = params(n, t, u, True, 0.2, 1)
        energy, theta = complete_flow_KUPCCGSD(p1,p2,p3,p4)

        aux = pd.DataFrame(energy)
        aux.to_csv("datos/FM/2KEnergyClose"+str(n)+"u"+str(u)+".csv")

        aux = pd.DataFrame(theta)
        aux.to_csv("datos/FM/2KThetaClose"+str(n)+"u"+str(u)+".csv")

        aux = [get_state_kUpCCGSD(angle, p3) for angle in theta]
        aux = pd.DataFrame(aux)
        aux.to_csv("datos/FM/2KStateClose"+str(n)+"u"+str(u)+".csv")


for n in N:
    for u in U:
        print(u)
        p1, p2, p3, p4 = params(n, t, u, True, 0.2, 1)
        energy, theta = complete_flow_KUPCCGSD(p1,p2,p3,p4)

        aux = pd.DataFrame(energy)
        aux.to_csv("datos/FM/3KEnergyClose"+str(n)+"u"+str(u)+".csv")

        aux = pd.DataFrame(theta)
        aux.to_csv("datos/FM/3KThetaClose"+str(n)+"u"+str(u)+".csv")

        aux = [get_state_kUpCCGSD(angle, p3) for angle in theta]
        aux = pd.DataFrame(aux)
        aux.to_csv("datos/FM/3KStateClose"+str(n)+"u"+str(u)+".csv")



for n in N:
    for u in U:
        print(u)
        p1, p2, p3, p4 = params(n, t, u, True, 0.2, 1)
        energy, theta = complete_flow_KUPCCGSD(p1,p2,p3,p4)

        aux = pd.DataFrame(energy)
        aux.to_csv("datos/FM/4KEnergyClose"+str(n)+"u"+str(u)+".csv")

        aux = pd.DataFrame(theta)
        aux.to_csv("datos/FM/4KThetaClose"+str(n)+"u"+str(u)+".csv")

        aux = [get_state_kUpCCGSD(angle, p3) for angle in theta]
        aux = pd.DataFrame(aux)
        aux.to_csv("datos/FM/4KStateClose"+str(n)+"u"+str(u)+".csv")