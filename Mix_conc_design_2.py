import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import tensorflow as tf
import pickle
import pandas as pd
from keras import backend
from sklearn.metrics import r2_score

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
def ANN_Sn():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32, input_dim=5, use_bias=True, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(16, use_bias=True, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(1))
    opt = tf.keras.optimizers.RMSprop(learning_rate=0.005, rho=0.9)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=[mae, rmse])
    return model
def pred_Sn(Mtx,XLC,Na2O,N_TX,Kd):
    model_ANN_Sn = ANN_Sn()
    model_ANN_Sn.load_weights('Save_Model/checkpoint_ANN_GA_Sn.hdf5')
    Scaler_Sn = pickle.load(open('Save_Model/scaler_ANN_GA_Sn.pkl', 'rb'))
    Xi = Scaler_Sn.transform(np.array((Mtx,XLC,Na2O,N_TX, Kd), dtype=float).reshape(-1,5))
    Sn = model_ANN_Sn(Xi).numpy()[0,0]
    return Sn
def pred_R(Mtx,XLC,Na2O,N_TX,Kd,tuoi=28):
    model_ANN_GA = ANN()
    model_ANN_GA.load_weights('Save_Model/checkpoint_ANN_GA.hdf5')
    Scaler_ANN_GA = pickle.load(open('Save_Model/scaler_ANN_GA.pkl', 'rb'))
    Xi = Scaler_ANN_GA.transform(np.array((Mtx,XLC,Na2O,N_TX, Kd, tuoi), dtype=float).reshape(-1,6))
    R = model_ANN_GA(Xi).numpy()[0,0]
    return R

def get_initial_Na2O_N_TX_XLC(R28):
    result=[]
    XLC_params = np.arange(30,61,5)              #7 gia tri
    Na2O_params = np.arange(4.0, 6.1, 0.1)    #21 gia tri => result gom 7*21=147 gia tri
    for XLC in XLC_params:
        for Na2O in Na2O_params:
            # From R28 = f(XLC, Na2O, N_TX) => a*N_TX^2 + b*N_TX + c = 0
            a = -159.665
            b = 31.2176 +18.7442*Na2O -1.81025*XLC
            c = - R28 -55.3333 +20.6822*Na2O +1.93365*XLC +0.0968158*Na2O*XLC -2.42788*(Na2O)**2 -0.0116982*(XLC)**2
            delta = b**2 - 4*a*c
            if delta<0:
                break
            else:
                N_TX = float((-b - np.sqrt(delta))/(2 * a))
                result.append([XLC, Na2O, N_TX])
    return result
def Mix_result(material_params, Mtx,XLC,Na2O,N_TX,Kd):
    Ms = material_params["Ms"]
    Na2O_ttl = material_params["Na2O_ttl"]
    SiO2_ttl = material_params["SiO2_ttl"]
    H2O_ttl = material_params["H2O_ttl"]
    gaNaOH = material_params["gaNaOH"]
    gaNa2SiO3 = material_params["gaNa2SiO3"]
    gaC = material_params["gaC"]
    g0C = material_params["g0C"]
    gaD = material_params["gaD"]
    g0D = material_params["g0D"]
    rD = material_params["rD"]
    gaTB = material_params["gaTB"]
    gaXLC = material_params["gaXLC"]

    C_price = material_params["C_price"]
    D_price = material_params["D_price"]
    XLC_price = material_params["XLC_price"]
    TB_price = material_params["TB_price"]
    Na2SiO3_price = material_params["Na2SiO3_price"]
    NaOH_price = material_params["NaOH_price"]
    N_price = material_params["N_price"]


    XLC = XLC/100
    Na2O = Na2O/100
    CKD_weight = Mtx/(1-Na2O*(1+Ms))
    XLC_weight = XLC*Mtx
    TB_weight = Mtx - XLC_weight
    NaOH_weight = 1.29*Na2O*CKD_weight*(1-Ms*Na2O_ttl/SiO2_ttl)
    Na2SiO3_weight = Ms*CKD_weight*Na2O/SiO2_ttl
    N = Mtx * N_TX
    N_weight = N - Na2SiO3_weight * H2O_ttl
    D_weight = 1000/(Kd*rD/g0D + 1/gaD)
    C_weight = (1000 - (TB_weight/gaTB + XLC_weight/gaXLC + NaOH_weight/gaNaOH + Na2SiO3_weight/gaNa2SiO3
                        + N_weight/1.0 + D_weight/gaD))*gaC

    cost = (C_weight/(g0C*1000) * C_price + D_weight/(g0D*1000) * D_price + TB_weight * TB_price +
            XLC_weight * XLC_price + NaOH_weight * NaOH_price + Na2SiO3_weight * Na2SiO3_price +
            N_weight * N_price)
    mix = {"cost":cost, "Mtx":Mtx, "%XLC":XLC*100, "%Na2O":Na2O*100, "N/TX":N_TX ,"Kd":Kd,
           "C_weight":C_weight, "D_weight": D_weight, "TB_weight": TB_weight,
           "XLC_weight":XLC_weight, "NaOH_weight":NaOH_weight, "Na2SiO3_weight":Na2SiO3_weight,
           "N_weight":N_weight}
    return mix

def main():
    material_params = {"Ms":1.2, "Na2O_ttl":0.0984, "SiO2_ttl":0.267, "H2O_ttl":0.6346,
                       "gaC":2.62,"g0C":1.420, "Mdl":2.6, "gaD":2.68, "g0D":1.432, "rD":0.47, "Dmax":20,
                       "gaTB":2.24, "gaXLC":2.85, "C_price":181.5, "D_price":242.0,
                       "XLC_price":500, "TB_price":100, "Na2SiO3_price":4000, "NaOH_price":13200,
                       "N_price":5, "gaNaOH":2.13, "gaNa2SiO3":1.45}
    R28 = 30
    Sn = 10
    initial_params = get_initial_Na2O_N_TX_XLC(R28)
    for x in initial_params:
        print(x)
    print('=============================')
    result = []
    for x in initial_params:
        XLC = x[0]
        Na2O = x[1]
        N_TX = x[2]
        N_params = np.arange(150, 181, 5)           #7 gia tri
        Kd_params = np.arange(1.50, 1.76, 0.05)     #6 gia tri => result gom 147*7*6 = 6174 gia tri
        for N in N_params:
            for Kd in Kd_params:
                Mtx = N/N_TX
                if 250<=Mtx<=450:
                    Sn_pred = pred_Sn(Mtx, XLC, Na2O, N_TX, Kd)
                    R28_pred = pred_R(Mtx, XLC, Na2O, N_TX, Kd)
                    result.append([Sn_pred, R28_pred, Mtx, XLC, Na2O, N_TX, Kd])
    for x in result:
        print(x)
    print('============================')
    mix_final = []
    for x in result:
        Sn_pred = x[0]
        R28_pred = x[1]
        err_Sn = (Sn_pred-Sn)/Sn*100
        err_R28 = (R28_pred-R28)/R28*100
        if (-5<err_Sn<5) and (0<err_R28<5):
            print(x)
            Mtx, XLC, Na2O, N_TX, Kd = x[2], x[3], x[4], x[5], x[6]
            mix_final.append(Mix_result(material_params, Mtx,XLC,Na2O,N_TX,Kd))
    print("+++++++++++++++++++++++++++++++++")
    for m in mix_final:
        print(m)

if __name__ == "__main__":
    main()