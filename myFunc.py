from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import numpy as np
import scipy.io as sio
import os
from scipy.signal import filtfilt

def SelectOptimizer(optimizer_sel, lr, param1, param2):   
    if optimizer_sel == 'Adam':
        opt = Adam(learning_rate = lr, 
                   beta_1 = param1, 
                   beta_2 = param2)
    if optimizer_sel == 'AMSgrad':
        opt = Adam(learning_rate = lr, 
                   beta_1 = param1, 
                   beta_2 = param2,
                   amsgrad = True)
    elif optimizer_sel == 'RMSprop':
        opt = RMSprop(learning_rate = lr, 
                      rho = param1, 
                      momentum = param2)
    elif optimizer_sel == 'SGD':
        opt = SGD(learning_rate = lr, 
                  momentum = param2)
    return opt

def LoadDataset(sub, filterType, dBdB, bm_C, b_T, m_T, T_base, T_shift, path):
    sub = path + sub[0:6] + 'VP' + sub[6:] + '.mat'
    mat_contents = sio.loadmat(sub)

    data_train = mat_contents['data'][0,0][0,0]
    data_test = mat_contents['data'][0,1][0,0]
    fs = data_train[2][0,0]

    if bm_C == 45:
        ch_keep = np.concatenate((np.arange(5,41,1), np.arange(42,51,1)))
    elif bm_C == 55:
        ch_keep = np.arange(5,60,1)

    X_train = data_train[0][:,ch_keep]
    X_test = data_test[0][:,ch_keep]
    tstim_train = data_train[4][0,]
    tstim_test = data_test[4][0,]
    samples_train = tstim_train.shape[0]
    samples_test = tstim_test.shape[0]
    
    # DETREND
    mean_X_train = np.mean(X_train, axis=0)
    X_detrended_train = X_train - mean_X_train
    mean_X_test = np.mean(X_test, axis=0)
    X_detrended_test = X_test - mean_X_test
    
    # FILTER
    dBdB = path + 'coeffFilt' + dBdB + '.mat'
    mat_contents = sio.loadmat(dBdB)
    filtCoeff = mat_contents
    if filterType == "f0":
        X_train = filtfilt(filtCoeff['bb'][0], filtCoeff['ab'][0], X_detrended_train, axis=0, method="gust")
        X_test = filtfilt(filtCoeff['bb'][0], filtCoeff['ab'][0], X_detrended_test, axis=0, method="gust")
    elif filterType == "f1":
        X_train = filtfilt(filtCoeff['b1'][0], filtCoeff['a1'][0], X_detrended_train, axis=0, method="gust")
        X_test = filtfilt(filtCoeff['b1'][0], filtCoeff['a1'][0], X_detrended_test, axis=0, method="gust")
    elif filterType == "f2":
        X_train = filtfilt(filtCoeff['b2'][0], filtCoeff['a2'][0], X_detrended_train, axis=0, method="gust")
        X_test = filtfilt(filtCoeff['b2'][0], filtCoeff['a2'][0], X_detrended_test, axis=0, method="gust")
    elif filterType == "f3":
        X_train = filtfilt(filtCoeff['be'][0], filtCoeff['ae'][0], X_detrended_train, axis=0, method="gust")
        X_test = filtfilt(filtCoeff['be'][0], filtCoeff['ae'][0], X_detrended_test, axis=0, method="gust")
    elif filterType == "f4":
        X_train = filtfilt(filtCoeff['bf'][0], 1, X_detrended_train, axis=0, method="gust")
        X_test = filtfilt(filtCoeff['bf'][0], 1, X_detrended_test, axis=0, method="gust")
    elif filterType == "f203":
        X_train = filtfilt(filtCoeff['b3'][0], filtCoeff['a3'][0], X_detrended_train, axis=0, method="gust")
        X_test = filtfilt(filtCoeff['b3'][0], filtCoeff['a3'][0], X_detrended_test, axis=0, method="gust")
    elif filterType == "f204":
        X_train = filtfilt(filtCoeff['b4'][0], filtCoeff['a4'][0], X_detrended_train, axis=0, method="gust")
        X_test = filtfilt(filtCoeff['b4'][0], filtCoeff['a4'][0], X_detrended_test, axis=0, method="gust")
    elif filterType == "f205":
        X_train = filtfilt(filtCoeff['b5'][0], filtCoeff['a5'][0], X_detrended_train, axis=0, method="gust")
        X_test = filtfilt(filtCoeff['b5'][0], filtCoeff['a5'][0], X_detrended_test, axis=0, method="gust")
    elif filterType == "f207":
        X_train = filtfilt(filtCoeff['b7'][0], filtCoeff['a7'][0], X_detrended_train, axis=0, method="gust")
        X_test = filtfilt(filtCoeff['b7'][0], filtCoeff['a7'][0], X_detrended_test, axis=0, method="gust")
    elif filterType == "f210":
        X_train = filtfilt(filtCoeff['b10'][0], filtCoeff['a10'][0], X_detrended_train, axis=0, method="gust")
        X_test = filtfilt(filtCoeff['b10'][0], filtCoeff['a10'][0], X_detrended_test, axis=0, method="gust")
    elif filterType == "f215":
        X_train = filtfilt(filtCoeff['b15'][0], filtCoeff['a15'][0], X_detrended_train, axis=0, method="gust")
        X_test = filtfilt(filtCoeff['b15'][0], filtCoeff['a15'][0], X_detrended_test, axis=0, method="gust")
    elif filterType == "f220":
        X_train = filtfilt(filtCoeff['b20'][0], filtCoeff['a20'][0], X_detrended_train, axis=0, method="gust")
        X_test = filtfilt(filtCoeff['b20'][0], filtCoeff['a20'][0], X_detrended_test, axis=0, method="gust")
    else:
        X_train = X_detrended_train
        X_test = X_detrended_test

    # TRAIN: TRASFORM IN A 3D MATRIX
    b_X3D_train = np.zeros((samples_train, bm_C, b_T))
    b_X3D_train_base = np.zeros((samples_train, bm_C, T_base))
    idx_samp = np.arange(T_shift, b_T - T_shift, 1)
    idx_samp_base = np.arange(T_shift - T_base, T_shift, 1)
    for samp in range(samples_train):
        arr1 = X_train[idx_samp + tstim_train[samp], :]
        b_X3D_train[samp, :, :] = arr1.transpose()
        arr2 = X_train[idx_samp_base + tstim_train[samp], :]
        b_X3D_train_base[samp, :, :] = arr2.transpose()
    # TRAIN: NORMALIZE
    baseline = np.mean(b_X3D_train_base, axis=2)
    b_X3D_train = b_X3D_train - baseline.reshape((samples_train, bm_C, 1))

    # TEST: TRASFORM IN A 3D MATRIX
    b_X3D_test = np.zeros((samples_test, bm_C, b_T))
    b_X3D_test_base = np.zeros((samples_test, bm_C, T_base))
    idx_samp = np.arange(T_shift, b_T - T_shift, 1)
    idx_samp_base = np.arange(T_shift - T_base, T_shift, 1)
    for samp in range(samples_test):
        arr1 = X_test[idx_samp + tstim_test[samp], :]
        b_X3D_test[samp, :, :] = arr1.transpose()
        arr2 = X_test[idx_samp_base + tstim_test[samp], :]
        b_X3D_test_base[samp, :, :] = arr2.transpose()
    # TEST: NORMALIZE
    baseline = np.mean(b_X3D_test_base, axis=2)
    b_X3D_test = b_X3D_test - baseline.reshape((samples_test, bm_C, 1))
    
    #########################
    #  MULTICLASS  #
    ####################
    samples_train = int(samples_train/6)
    samples_test = int(samples_test/6)

    # TRAIN: TRASFORM IN A 3D MATRIX
    m_X3D_train = np.zeros((samples_train, bm_C, m_T))
    m_X3D_train_base = np.zeros((samples_train, bm_C, T_base))
    idx_samp = np.arange(T_shift, m_T - T_shift, 1)
    idx_samp_base = np.arange(T_shift - T_base, T_shift, 1)
    for samp in range(samples_train):
        arr1 = X_train[idx_samp + tstim_train[samp*6], :]
        m_X3D_train[samp, :, :] = arr1.transpose()
        arr2 = X_train[idx_samp_base + tstim_train[samp*6], :]
        m_X3D_train_base[samp, :, :] = arr2.transpose()
    # TRAIN: NORMALIZE
    baseline = np.mean(m_X3D_train_base, axis=2)
    m_X3D_train = m_X3D_train - baseline.reshape((samples_train, bm_C, 1))

    # TEST: TRASFORM IN A 3D MATRIX
    m_X3D_test = np.zeros((samples_test, bm_C, m_T))
    m_X3D_test_base = np.zeros((samples_test, bm_C, T_base))
    idx_samp = np.arange(T_shift, m_T - T_shift, 1)
    idx_samp_base = np.arange(T_shift - T_base, T_shift, 1)
    for samp in range(samples_test):
        arr1 = X_test[idx_samp + tstim_test[samp*6], :]
        m_X3D_test[samp, :, :] = arr1.transpose()
        arr2 = X_test[idx_samp_base + tstim_test[samp*6], :]
        m_X3D_test_base[samp, :, :] = arr2.transpose()
    # TEST: NORMALIZE
    baseline = np.mean(m_X3D_test_base, axis=2)
    m_X3D_test = m_X3D_test - baseline.reshape((samples_test, bm_C, 1))

    return b_X3D_train, b_X3D_test, m_X3D_train, m_X3D_test