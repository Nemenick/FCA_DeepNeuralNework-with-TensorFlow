from Ising2d import Termalizza  # funzione che mi termalizza 1 sola configurazione in 2D

import numpy as np
from matplotlib import pyplot as plt
import tensorflow
from tensorflow import keras
from keras.layers import Dense # type: ignore
from keras import optimizers

from keras.callbacks import EarlyStopping # type: ignore


def imparo_ising(n=35, ntrain=18, nval=10, ntest=4, momento=0.8, read_file=True):
    """
    Neural network made up of only FullyConnected layers (or Dense layers)
    The TensorFlow implementation is very fast during training and it is very easy to generalize

    a name of an example file accepted is 35conf4.dat
    Here 35 corresponds to the number of spin per dimension (35x35)
    and 4 corresponds to ntrain (or ntest or nval...) and it is the number of Temperature used.
    For each temperature I sample 20 configuration, so in 35conf4.dat i have 4x20=80 spin conf. (1 conf. is 1 raw)
    TODO read_file = True only if alredy generated my config.: (n)conf(ntrain).dat and (n)conf(ntest).dat alredy present
    TODO if readfile = false at first I generate my configurations (it takes a while..)
    """

    """############################################## Generate my Data ##############################################"""
    # n = 35                      # FIXME (ATTENTION! it has to be fixed to decide what file to read!)
    data, data_val, data_test = [], [], []
    y, y_val, y_test = [], [], []
    if not read_file:
        # In this section I generate my data rather than loading them
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
        print("Ho generato tutte le \"training\" configurazioni")
        data = np.array(data)

        # Now i generate validation data
        t = [0.1 + 3 / nval * i for i in range(nval // 2)] + [4 + 40 / nval * i for i in range(nval // 2)]
        tc = 2 / np.log(1 + np.sqrt(2))
        for temp in t:
            spin = Termalizza(temp, L=n)
            for _ in range(10):
                if temp < tc:
                    y_val.append(0)
                else:
                    y_val.append(1)
                data_val.append([])
                Termalizza(temp, L=n, spin=spin, equilibrato=True)
                for i in range(n):
                    for j in range(n):
                        data_val[-1].append(spin[i + 1][j + 1])
        print("Ho generato tutte le \"training\" configurazioni")
        data_val = np.array(data_val)

        # Now i generate my test data
        t = [0.1 + 3 / nval * i for i in range(nval // 2)] + [4 + 40 / nval * i for i in range(nval // 2)]
        tc = 2 / np.log(1 + np.sqrt(2))
        for temp in t:
            spin = Termalizza(temp, L=n)
            for _ in range(10):
                if temp < tc:
                    y_test.append(0)
                else:
                    y_test.append(1)
                data_test.append([])
                Termalizza(temp, L=n, spin=spin, equilibrato=True)
                for i in range(n):
                    for j in range(n):
                        data_test[-1].append(spin[i + 1][j + 1])
        print("Ho generato tutte le \"training\" configurazioni")
        data_test = np.array(data_test)

    else:
        # if read_file I load my data than generating them! (files have to exist before I lunch this section)
        file = open(str(n) + "conf" + str(ntrain) + ".dat", "r")
        file_y = open(str(n) + "conf" + str(ntrain) + "y.dat", "r")
        data = np.loadtxt(file)
        y = np.loadtxt(file_y)

        file = open(str(n) + "conf" + str(nval) + ".dat", "r")
        file_y = open(str(n) + "conf" + str(nval) + "y.dat", "r")
        data_val = np.loadtxt(file)
        y_val = np.loadtxt(file_y)

        file = open(str(n) + "conf" + str(ntest) + ".dat", "r")
        file_y = open(str(n) + "conf" + str(ntest) + "y.dat", "r")
        data_test = np.loadtxt(file)
        y_test = np.loadtxt(file_y)
        file.close()

    print(data.shape)

    """####################################### Build and train my network ########################################"""
    # change the hidden layer size and study the effect [(2), (5), (10), (25)]
    # or add more layers. For example add the line "Dense(3, activation="relu")," after the first Dense layer
    rete = tensorflow.keras.models.Sequential([
        Dense(10, input_shape=(data.shape[1],), activation="sigmoid"),
        Dense(5, activation="sigmoid"),
        Dense(1, activation="sigmoid")
    ])
    rete.compile(

        optimizer=optimizers.SGD(momentum=momento),
        loss="binary_crossentropy",
        metrics=['accuracy']
    )
    rete.summary()
    # Start my training
    batches, epoche, pazienza = 32, 20, 5
    storia = rete.fit(data, y,
                      batch_size=batches,
                      epochs=epoche,
                      validation_data=(data_val, y_val),
                      callbacks=EarlyStopping(patience=pazienza,  restore_best_weights=True))
    print(storia.history.keys())

    """################################################### Test ####################################################"""
    yp_test = rete.predict(data_test)                       # make my predictions
    yp_ok_test = []
    for i in yp_test:
        yp_ok_test.append(i[0])
    yp_ok_test = np.array(yp_ok_test)

    total_predictions = len(y_test)
    correct_predictions = 0
    erroneous_predictions = 0                               # calculating the total, erroneous and correct predictions
    for i in range(total_predictions):
        if abs(y_test[i] - yp_ok_test[i]) < 0.5:
            correct_predictions = correct_predictions + 1
        else:
            erroneous_predictions = erroneous_predictions + 1
    print("test predictions:\ntotal/correct/erroneous:\n", total_predictions, "/", correct_predictions, "/", erroneous_predictions, "/")

    print("test Accuracy= ", correct_predictions / total_predictions)

    """################################################### Plot ####################################################"""
    loss_train = storia.history["loss"]
    loss_val = storia.history["val_loss"]
    acc_train = storia.history["accuracy"]
    acc_val = storia.history["val_accuracy"]

    plt.plot(range(len(acc_train)), acc_train, label="acc_train")
    plt.plot(range(len(acc_val)), acc_val, label="acc_val")
    plt.legend()
    plt.savefig("Accuracy_Ising2d")
    plt.show()
    plt.clf()

    plt.yscale("log")
    plt.plot(range(len(loss_train)), loss_train, label="loss_train")
    plt.plot(range(len(loss_val)), loss_val, label="loss_val")
    plt.legend()
    plt.savefig("Loss_Ising2d")
    plt.show()
    plt.clf()


imparo_ising(n=35, ntrain=14, nval=12, ntest=10, momento=0.8, read_file=True)
