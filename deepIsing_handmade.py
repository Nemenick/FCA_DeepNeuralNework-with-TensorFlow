from Ising2d import Termalizza  # funzione che mi termalizza 1 sola configurazione in 2D
import numpy as np


def sigma(x):
    return 1 / (np.exp(-x) + 1)


def sigma_diff(x):
    return sigma(x) * (1 - sigma(x))


def imparo_ising(n=35, ntrain=10, ntest=14, precision=10 ** (-0.5), dt=10 ** (-1.5), gamma=0.8, read_file=True):
    """
    This is the Python implementation of the network with one hidden layer (containing two hidden neurons)
    and one output layer. The network aims to predict if an Ising configuration is above or under tc

    A name of an example file accepted is 35conf4.dat
    Here 35 corresponds to the number of spin per dimension (35x35)
    and 4 corresponds to ntrain (or ntest) and it is the number of Temperature used.
    For each temperature I sample 20 configuration, so in 35conf4.dat i have 4x20=80 spin conf. (1 conf. is 1 raw)
    TODO read_file = True only if alredy generated my config.: (n)conf(ntrain).dat and (n)conf(ntest).dat alredy present
    TODO if readfile == False at first I generate my configurations (it takes a while...)
    """

    """############################################## Generate my Data ##############################################"""
    # n = 35          # Number Spin for dimension (2D)
    precision_dt = precision * dt
    data = []
    y = []
    if not read_file:
        # abbiamo 10*ntrain configurazioni, la metà sopra T_c e la metà sotto
        t = [0.1 + 3 / ntrain * i for i in range(ntrain // 2)] + [4 + 40 / ntrain * i for i in range(ntrain // 2)]
        tc = 2 / np.log(1 + np.sqrt(2))
        for temp in t:
            spin = Termalizza(temp, L=n)
            for _ in range(10):
                if temp < tc:
                    y.append(0)
                else:
                    y.append(1)
                data.append([])
                Termalizza(temp, L=n, spin=spin, equilibrato=True)
                for i in range(n):
                    for j in range(n):
                        data[-1].append(spin[i + 1][j + 1])
        print("Ho generato tutte le configurazioni")
        data = np.array(data)
    else:
        file = open(str(n) + "conf" + str(ntrain) + ".dat", "r")
        file_y = open(str(n) + "conf" + str(ntrain) + "y.dat", "r")
        # file = open("50only2temp.dat", "r")
        data = np.loadtxt(file)
        y = np.loadtxt(file_y)

    """####################################### Build and train my network ########################################"""
    t = 0                   # Time
    cold = 0                # Loss function
    cnew = np.infty
    b2 = 1
    b11 = 1                 # parameters
    b12 = 1
    theta21 = theta22 = 0.5
    theta1 = [10. / n ** 2 for _ in range(n ** 2)]
    theta2 = [-10. / n ** 2 for _ in range(n ** 2)]

    g_b2 = g_b11 = g_b12 = 0
    g_theta21 = g_theta22 = 0                   # gradients
    g_theta1 = [0 for _ in range(n ** 2)]
    g_theta2 = [0 for _ in range(n ** 2)]
    """ # start training """
    while np.abs((cold - cnew)) > precision_dt:
        t += 1
        cold = cnew
        cnew = 0
        g_b2 = gamma * g_b2
        g_b11 = gamma * g_b11
        g_b12 = gamma * g_b12
        g_theta21 = gamma * g_theta21
        g_theta22 = gamma * g_theta22
        g_theta1 = [gamma * g_theta1[i] for i in range(n ** 2)]
        g_theta2 = [gamma * g_theta2[i] for i in range(n ** 2)]

        for nconf in range(len(data)):
            ############## Forward ##############
            z11 = np.sum(np.array(theta1) * data[nconf]) + b11
            z12 = np.sum(np.array(theta2) * data[nconf]) + b12
            a11 = sigma(z11)
            a12 = sigma(z12)
            z2 = b2 + a11 * theta21 + a12 * theta22
            a2 = sigma(z2)

            ############## Backward ##############
            delta2 = a2 - y[nconf]
            delta11 = sigma_diff(z11) * delta2 * theta21
            delta12 = sigma_diff(z12) * delta2 * theta22

            g_b2 += delta2 * dt
            g_b11 += delta11 * dt
            g_b12 += delta12 * dt
            g_theta21 += delta2 * a11 * dt
            g_theta22 += delta2 * a12 * dt
            for i in range(n ** 2):
                g_theta1[i] += data[nconf][i] * delta11 * dt
                g_theta2[i] += data[nconf][i] * delta12 * dt
            cnew -= y[nconf] * np.log(a2) + (1 - y[nconf]) * np.log(1 - a2)

        ############## Weights update ##############
        b2 -= g_b2
        b11 -= g_b11
        b12 -= g_b12
        theta21 -= g_theta21
        theta22 -= g_theta22
        for i in range(n ** 2):
            theta1[i] -= g_theta1[i]
            theta2[i] -= g_theta2[i]
        print("b2 ", b2, "\tb12 ", b12, "\ntheta21 ", theta21, "\ttheta1[1] ", theta1[1])
        print("ho finito il ciclo", t)
    accuratezza = 0
    print("\n")

    """################################### Predictions ####################################"""
    ############## Predictions on Training set ##############
    for nconf in range(len(data)):
        z11 = np.sum(np.array(theta1) * data[nconf]) + b11
        z12 = np.sum(np.array(theta2) * data[nconf]) + b12
        a11 = sigma(z11)
        a12 = sigma(z12)
        z2 = b2 + a11 * theta21 + a12 * theta22
        a2 = sigma(z2)
        if abs(y[nconf] - a2) < 0.5:
            accuratezza += 1
        # print("y=", y[nconf], "output=", a2)
    print("\naccuracy training set= ", accuratezza / len(data))

    ############## Predictions on Test set ##############
    file = open(str(n) + "conf" + str(ntest) + ".dat", "r")
    file_y = open(str(n) + "conf" + str(ntest) + "y.dat", "r")
    data_test = np.loadtxt(file)
    y_test = np.loadtxt(file_y)

    print("\n")

    accuratezza = 0
    for nconf in range(len(data_test)):
        z11 = np.sum(np.array(theta1) * data_test[nconf]) + b11
        z12 = np.sum(np.array(theta2) * data_test[nconf]) + b12
        a11 = sigma(z11)
        a12 = sigma(z12)
        z2 = b2 + a11 * theta21 + a12 * theta22
        a2 = sigma(z2)
        if abs(y_test[nconf] - a2) < 0.5:
            accuratezza += 1
        # print("y=", y_test[nconf], "output=", a2)
    print("\naccuracy test set= ", accuratezza / len(data_test))
    print("b2,b12,b11,theta21\ntheta22,theta1[2],theta2[2]")
    print(b2, b12, b11, theta21, "\n", theta22, theta1[2], theta2[2])


imparo_ising()
