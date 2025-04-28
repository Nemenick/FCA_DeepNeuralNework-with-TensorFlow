from Ising2d import Termalizza             # 1  configuration in Ising2D
import numpy as np

"""
Code that generates some Ising2D configurations at equilibrium
"""
N = 35
n_temp = 0  # number of temperatures used to generate my data
config_per_temp = 20 # number of configuration for single temperature
file = open(str(N) + "conf" + str(n_temp) + ".dat", "w")
filey = open(str(N) + "conf" + str(n_temp) + "y.dat", "w")

y = []
T = [0.5 + 3.5 / n_temp * i for i in range(n_temp // 2)] + [4 + 20 / n_temp * i for i in range(n_temp // 2)]
# T = [0.5, 50]
Tc = 2/np.log(1+np.sqrt(2))
for temp in T:
    spin = Termalizza(temp, L=N)
    for _ in range(config_per_temp):
        if temp < Tc:
            y.append(0)
        else:
            y.append(1)
        Termalizza(temp, L=N, spin=spin, equilibrato=True)
        # In the previous line I decided to put equilibrato=True because I start from the current configuration and I do
        # some few metropolis steps to generate a new configuration

        for i in range(N):
            for j in range(N):
                sspin = (spin[i + 1][j + 1])  # così gli spin sono 0 e 1 sul file
                file.write("%d " % sspin)

        file.write("\n")
for i in range(len(y)):
    filey.write("%d " % y[i])

print("Ho generato tutte le configurazioni")