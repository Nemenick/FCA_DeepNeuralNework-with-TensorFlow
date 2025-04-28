from random import choice
from numpy import e
import numpy as np
import matplotlib.pyplot as plt


def Termalizza(T, L=10, spin=None, equilibrato=False, precision=10 ** (-2), stepsmax=None):
    """
    This function generates a spin configuration in a 2D Ising model equilibrated at temperature T
    I use the Born-von Karman boundary conditions (the first raw(or col) interacts with the last one)
    For this reason the shape of the "spin" variable is (L+2xL+2) and not only (LxL)


    L:              Number of spin per dimension (LxL)
    T:              Temperature of termalizzation
    spin:           Initial spin configuration (if passed)
    equilibrato:    Bool, if the configuration is alredy at equilibrium (useful: doing few metropolis steps generate a new config)
    precision:      Precision in the Energy of the configuration, decides when to stop the algorithm
    stepsmax:
    Tc:             Critical temperature
    NOTE  T=2.35 magnetizzation is 1/3 (on 100 spin), for T=2.4 m=0.07
    """
    Tc = 2.269185
    if spin is None:
        equilibrato = False
        spin = [[1 for _ in range(L + 2)] for __ in range(L + 2)]     # L è numero spin,il resto è per condizioni BVK
        if T < Tc:
            a = choice([-1, 1])
            print("gli spin adesso sono tutti ", a)
            for i in range(L+2):
                for j in range(L+2):
                    spin[i][j] = a
        else:
            print("gli spin adesso sono tutti a caso ")
            for i in range(L):
                for j in range(L):
                    spin[i + 1][j + 1] = choice([-1, 1])
            for i in range(1, L+1):
                spin[0][i] = spin[L][i]
                spin[i][0] = spin[i][L]
                spin[L + 1][i] = spin[1][i]
                spin[i][L + 1] = spin[i][1]
    else:
        L = len(spin[0])-2

    if stepsmax is None:
        stepsmax = 100

    if equilibrato:
        step = 0
        while step < (L*L*10):
            metropolis(spin, L, T)
            step += 1

    else:
        E = 0
        for i in range(1, L+1):
            for j in range(1, L+1):
                E -= spin[i][j] * (spin[i + 1][j] + spin[i - 1][j] + spin[i][j + 1] + spin[i][j - 1])/2
        i = 0
        while i < stepsmax:
            i += 1
            DE = 0
            for j in range(L*L):
                DE += metropolis(spin, L, T)
            if abs(DE/L*L) < precision:         # TODO  controlla il discostamento tra media ultimi 100 E e ultime 200 E
                print("exit at step ",i)
                i = stepsmax
            E += DE
    if not equilibrato:
        return spin


def metropolis(spin, L, T):
    """ Makes 1 step of metropolis algorithm"""
    i = choice(range(L)) + 1        # choose 1 raw
    j = choice(range(L)) + 1        # choose 1 column
    DE = 2 * spin[i][j] * (spin[i + 1][j] + spin[i - 1][j] + spin[i][j + 1] + spin[i][j - 1])
    prob = e ** (-DE / T)
    if DE < 0 or prob > np.random.uniform(0, 1):        # evolve the ij-th spin
        spin[i][j] *= -1
        if i == L:
            spin[0][j] = spin[i][j]
        if j == L:
            spin[i][0] = spin[i][j]
        if i == 1:
            spin[L+1][j] = spin[i][j]
        if j == 1:
            spin[i][L+1] = spin[i][j]
    return DE


def energia(spin,L):
    """ Calculating the energy of the spin config."""
    E=0
    for i in range(1, L + 1):
        for j in range(1, L + 1):
            E -= spin[i][j] * (spin[i + 1][j] + spin[i - 1][j] + spin[i][j + 1] + spin[i][j - 1]) / 2
    return E


def Magneto(T):
    """ Calculating the Magnetization of the spin config."""
    z = e**(-2/T)
    return ((1+z**2)**0.25*(1-6*z**2+z**4)**0.125)/(1-z**2)**0.5


if __name__ == '__main__':
    Tc = 2.269185
    L = 300
    Cv_T = []
    Nconf_T = 30
    T = 0
    M_T = []        # Questa è M in funzione di T
    E_T = []
    T_T = []        # temperature usate
    for _ in range(50):
        T += 0.1
        T_T.append(T)
        M = []
        E = []
        spin = Termalizza(T, L, precision=10 ** (-4))
        for __ in range(Nconf_T):
            Termalizza(T, L, precision=10 ** (-4), spin=spin, equilibrato=True)
            m = 0                       # magnetizzazione di ciascuna conf.
            for i in range(1, L + 1):
                for j in range(1, L+1):
                    m += spin[i][j]
            E.append(energia(spin, L))
            M.append(m/(L*L))
            # TODO statistica
        E_T.append(sum(E)/(Nconf_T))
        Cv = np.sum((np.array(E)**2-(E_T[-1])**2)) / (L ** 4 * T ** 2)
        Cv_T.append(Cv)
        M = sum(M)/Nconf_T
        M_T.append(abs(M))
    step = Tc/50
    tt = [(i+1)*step for i in range(49)]
    plt.plot(tt, Magneto(np.array(tt)), linewidth="3", label="Esatto")
    plt.plot(T_T, M_T, color="purple", label="Simulazione")
    plt.xlabel("Temperatura")
    plt.ylabel("Magnetizzazione")
    plt.plot([2.269185,2.269185],[0,1], color="orange")
    plt.annotate("$T_c$", xy=(2.269185,0.0))
    plt.legend()
    plt.savefig("Magneto")
    plt.show()

    E_T = np.array(E_T)/(L**2)
    plt.plot(T_T, E_T, color="green")
    plt.xlabel("Temperatura")
    plt.ylabel("Energia")
    plt.plot([2.269185, 2.269185], [min(E_T), max(E_T)],color="orange")
    plt.annotate("$T_c$", xy=(2.269185, min(E_T)))
    plt.savefig("EnergiaPerSpin")
    plt.show()

    plt.plot(T_T, Cv_T, color="red")
    plt.xlabel("Temperatura")
    plt.ylabel("Calore specifico")
    plt.plot([2.269185, 2.269185], [min(Cv_T), max(Cv_T)], color="orange")
    plt.annotate("$T_c$", xy=(2.269185, min(Cv_T)))
    plt.savefig("CvPerSpin")
    plt.show()


    # spin = Termalizza(T, L, precision=10 ** (2))
    # m=0
    # for i in range(1, L+1):
    #     for j in range(1, L+1):
    #         # print(spin[i][j], end=" ")
    #         m+=spin[i][j]
    #     #print("\n")
    # m=m/(L*L)
    # print("magnetizzazione=", m)
    # Termalizza(T, L, precision=10 ** (2), spin=spin, equilibrato=True)
    # m = 0
    # for i in range(1, L + 1):
    #     for j in range(1, L + 1):
    #         # print(spin[i][j], end=" ")
    #         m += spin[i][j]
    #     # print("\n")
    # m = m / (L * L)
    # print("magnetizzazione=", m)
