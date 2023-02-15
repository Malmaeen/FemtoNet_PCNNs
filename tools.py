import pandas as pd
import numpy as np
import os,sys
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer


def data_loader(polarization=1):
    df1 = pd.read_csv ('/Data/defurne_e00_110_dvcs.csv', header = None)
    df2 = pd.read_csv ('/Data/defurne_e07_007_dvcs.csv', header = None)
    df3 = pd.read_csv ('/Data/georges_e12_114_dvcs.csv', header = None)
    df4 = pd.read_csv ('/Data/jo_e01dvcs.csv', header = None)
    A1 = df1.to_numpy()
    A1 = A1[:, :8]
    A2 = df2.to_numpy()
    A3 = df3.to_numpy()
    A3 = A3[:, :8]
    A4 = df4.to_numpy()
    A = np.concatenate([A1, A2, A3, A4], axis = 0)
    index = np.where(A[:, 5] == polarization)
    A = A[index]
    test = A[:24, :]
    train = A[24:, :]
    ###### train #####
    param = train[:, :5]
    sigma = train[:, 6]
    sigma_err = train[:, 7]
    ###### test #####
    param_test = test[:, :5]
    sigma_test = test[:, 6]
    sigma_err_test = test[:, 7]
    ###### Data Augmentation #####
    DUP = 10
    param_rep = np.repeat(param, DUP, axis = 0)
    sigma_rep = np.repeat(sigma, DUP, axis = 0)
    sigma_err_rep = np.repeat(sigma_err, DUP, axis = 0)
    gaussian = np.random.normal(0, 1, [param_rep.shape[0]])
    sigma_rep = sigma_rep + sigma_err_rep*gaussian
    sigma_rep= sigma_rep.reshape(param_rep.shape[0],1)
    return param, param_rep,param_test, sigma, sigma_rep, sigma_test
    
def checkdir(path):
            if not os.path.exists(path): 
                os.makedirs(path)
def Ex_test():
    phi = [7.5, 22.5, 37.5, 52.5, 67.5, 82.5, 97.5, 112.5, 127.5, 142.5, 157.5, 172.5,
           187.5, 202.5, 217.5, 232.5, 247.5, 262.5, 277.5, 292.5, 307.5, 322.5, 337.5, 352.5]
    phi=np.array(phi).reshape(-1,1)
    xbj_= np.full((24,1),0.365)
    t_= np.full((24,1),-0.2)
    Q2_= np.full((24,1),2.0)
    k0_= np.full((24,1),5.75)
    test_example= np.concatenate([xbj_,t_,Q2_,k0_,phi],axis=1)
    return test_example

