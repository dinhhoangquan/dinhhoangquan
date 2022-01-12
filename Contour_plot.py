import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import tensorflow as tf
import pickle
import pandas as pd
from keras import backend
from sklearn.metrics import r2_score

def pred_Kd(Mtx,XLC,Na2O,N_TX,Sn):
    model_RFR_Kd = pickle.load(open('Save_Model/RFR_Kd.sav', 'rb'))
    Scaler_Kd = pickle.load(open('Save_Model/scaler_Kd.pkl', 'rb'))
    Xi = np.array((Mtx,XLC,Na2O,N_TX, Sn), dtype=object).reshape(-1, 5)
    Xi = Scaler_Kd.transform(Xi)
    Kd = model_RFR_Kd.predict(Xi)
    return Kd
def pred_Sn(Mtx,XLC,Na2O,N_TX,Kd):
    model_RFR_Sn = pickle.load(open('Save_Model/RFR_Sn.sav', 'rb'))
    Scaler_Sn = pickle.load(open('Save_Model/scaler_Sn.pkl', 'rb'))
    Xi = np.array((Mtx,XLC,Na2O,N_TX, Kd), dtype=object).reshape(-1, 5)
    Xi = Scaler_Sn.transform(Xi)
    Sn = model_RFR_Sn.predict(Xi)
    return Sn

def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))
def mae(y_true, y_pred):
    return backend.mean(backend.abs(y_pred - y_true), axis=-1)
def ANN():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32, input_dim=6, use_bias=True, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(16, use_bias=True, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(1))
    opt = tf.keras.optimizers.RMSprop(learning_rate=0.005, rho=0.9)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=[mae, rmse])
    return model
def pred_R(Mtx,XLC,Na2O=5,N_TX=0.4,Kd=1.635,tuoi=28):
    model_ANN_GA = ANN()
    model_ANN_GA.load_weights('Save_Model/checkpoint_ANN_GA.hdf5')
    Scaler_ANN_GA = pickle.load(open('Save_Model/scaler_ANN_GA.pkl', 'rb'))
    X = np.array((Mtx,XLC,Na2O,N_TX, Kd, tuoi), dtype=object).reshape(-1, 6)
    X = Scaler_ANN_GA.transform(X)
    R = model_ANN_GA.predict(X)
    return R

def main():
    # # Mtx ~ XLC contour plot
    # Mtx = np.arange(250, 451, 5)
    # XLC = np.arange(30, 61, 5)
    # R28 = []
    # for xlc in XLC:
    #     for mtx in Mtx:
    #         r28 = pred_R(mtx, xlc, 5, 0.4, 1.635, 28)
    #         R28.append(r28[0,0])
    # x, y = np.meshgrid(Mtx, XLC)
    # z = np.array(R28).reshape(x.shape[0],x.shape[1])
    # fig1, ax1 = plt.subplots()
    # CS1 = ax1.contour(x, y, z)
    # ax1.clabel(CS1, CS1.levels, inline=True, fontsize=10)
    # # fig1.colorbar(CS1)  # Add a colorbar to a plot
    # ax1.set_title('%Na20=5%, N/TX=0.4, Kd=1.635, tuoi=28')
    # ax1.set_xlabel('Mtx (kg)')
    # ax1.set_ylabel('%XLC (%)')
    # plt.savefig('Save_png/12.png', format='png', dpi=300)
    # plt.show()
    #
    # # Mtx ~ N/TX contour plot
    # Mtx = np.arange(250, 451, 5)
    # N_TX = np.arange(0.355, 0.641, 0.005)
    # R28 = []
    # for n_tx in N_TX:
    #     for mtx in Mtx:
    #         r28 = pred_R(mtx, 50, 5, n_tx, 1.635, 28)
    #         R28.append(r28[0, 0])
    # x, y = np.meshgrid(Mtx, N_TX)
    # z = np.array(R28).reshape(x.shape[0], x.shape[1])
    # fig1, ax1 = plt.subplots()
    # CS1 = ax1.contour(x, y, z)
    # ax1.clabel(CS1, CS1.levels, inline=True, fontsize=10)
    # # fig1.colorbar(CS1)  # Add a colorbar to a plot
    # ax1.set_title('%XLC=50%, %Na20=5%, Kd=1.635, tuoi=28')
    # ax1.set_xlabel('Mtx (kg)')
    # ax1.set_ylabel('N/TX')
    # plt.savefig('Save_png/13.png', format='png', dpi=300)
    # plt.show()

    # # Mtx ~ Kd contour plot
    # Mtx = np.arange(250, 451, 5)
    # Kd = np.arange(1.550, 1.756, 0.005)
    # R28 = []
    # for kd in Kd:
    #     for mtx in Mtx:
    #         r28 = pred_R(mtx, 50, 5, 0.45, kd, 28)
    #         R28.append(r28[0, 0])
    # x, y = np.meshgrid(Mtx, Kd)
    # z = np.array(R28).reshape(x.shape[0], x.shape[1])
    # fig1, ax1 = plt.subplots()
    # CS1 = ax1.contour(x, y, z)
    # ax1.clabel(CS1, CS1.levels, inline=True, fontsize=10)
    # # fig1.colorbar(CS1)  # Add a colorbar to a plot
    # ax1.set_title('%XLC=50%, %Na20=5%, N/TX=0.45, tuoi=28')
    # ax1.set_xlabel('Mtx (kg)')
    # ax1.set_ylabel('Kd')
    # plt.savefig('Save_png/14.png', format='png', dpi=300)
    # plt.show()

    # Mtx ~ Kd - Sn contour plot
    Mtx = np.arange(250, 451, 5)
    Kd = np.arange(1.550, 1.756, 0.05)
    R28 = []
    for kd in Kd:
        for mtx in Mtx:
            sn = pred_Sn(mtx, 50, 5, 0.45, kd)
            R28.append(sn)
    x, y = np.meshgrid(Mtx, Kd)
    z = np.array(R28).reshape(x.shape[0], x.shape[1])
    fig1, ax1 = plt.subplots()
    CS1 = ax1.contour(x, y, z)
    ax1.clabel(CS1, CS1.levels, inline=True, fontsize=10)
    # fig1.colorbar(CS1)  # Add a colorbar to a plot
    ax1.set_title('Sn - %XLC=50%, %Na20=5%, N/TX=0.45')
    ax1.set_xlabel('Mtx (kg)')
    ax1.set_ylabel('Kd')
    plt.savefig('Save_png/15.png', format='png', dpi=300)
    plt.show()
    #
    # # Mtx ~ N_TX - Sn contour plot
    # Mtx = np.arange(250, 451, 5)
    # N_TX = np.arange(0.355, 0.641, 0.01)
    # R28 = []
    # for n_tx in N_TX:
    #     for mtx in Mtx:
    #         sn = pred_Sn(mtx, 50, 5, n_tx, 1.635)
    #         R28.append(sn)
    # x, y = np.meshgrid(Mtx, N_TX)
    # z = np.array(R28).reshape(x.shape[0], x.shape[1])
    # fig1, ax1 = plt.subplots()
    # CS1 = ax1.contour(x, y, z)
    # ax1.clabel(CS1, CS1.levels, inline=True, fontsize=10)
    # # fig1.colorbar(CS1)  # Add a colorbar to a plot
    # ax1.set_title('Sn - %XLC=50%, %Na20=5%, N/TX=0.45')
    # ax1.set_xlabel('Mtx (kg)')
    # ax1.set_ylabel('N/TX')
    # plt.savefig('Save_png/16.png', format='png', dpi=300)
    # plt.show()
    #
    # # Kd ~ N_TX - Sn contour plot
    # Kd = np.arange(1.550, 1.756, 0.05)
    # N_TX = np.arange(0.355, 0.641, 0.01)
    # R28 = []
    # for n_tx in N_TX:
    #     for kd in Kd:
    #         sn = pred_Sn(350, 50, 5, n_tx, kd)
    #         R28.append(sn)
    # x, y = np.meshgrid(Kd, N_TX)
    # z = np.array(R28).reshape(x.shape[0], x.shape[1])
    # fig1, ax1 = plt.subplots()
    # CS1 = ax1.contour(x, y, z)
    # ax1.clabel(CS1, CS1.levels, inline=True, fontsize=10)
    # # fig1.colorbar(CS1)  # Add a colorbar to a plot
    # ax1.set_title('Sn - Mtx = 350kg, %XLC=50%, %Na20=5%')
    # ax1.set_xlabel('Kd')
    # ax1.set_ylabel('N/TX')
    # plt.savefig('Save_png/17.png', format='png', dpi=300)
    # plt.show()

    # Mtx ~ Sn - Kd contour plot
    Mtx = np.arange(250, 451, 5)
    Sn = np.arange(5, 25, 1)
    Kd = []
    for sn in Sn:
        for mtx in Mtx:
            kd = pred_Kd(mtx, 50, 5, 0.4, sn)
            Kd.append(kd)
    x, y = np.meshgrid(Mtx, Sn)
    z = np.array(Kd).reshape(x.shape[0], x.shape[1])
    fig1, ax1 = plt.subplots()
    CS1 = ax1.contour(x, y, z)
    ax1.clabel(CS1, CS1.levels, inline=True, fontsize=10)
    # fig1.colorbar(CS1)  # Add a colorbar to a plot
    ax1.set_title('Kd - %XLC=50%, %Na20=5%, N/TX=0.5')
    ax1.set_xlabel('Mtx (kg)')
    ax1.set_ylabel('Sn')
    plt.savefig('Save_png/18.png', format='png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()