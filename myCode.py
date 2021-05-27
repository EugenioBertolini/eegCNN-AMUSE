# from google.colab import drive
# drive.mount('/content/drive')

import numpy as np
import scipy.io as sio
import seaborn as sns
import os
import sys
sys.path.insert(0,'/content/drive/MyDrive/GoogleColab') # set the path for your import

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.random import set_seed
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')

path0 = '/content/drive/MyDrive/GoogleColab/' # set your path for your files
from myCNN import EEGNet, ReturnBestEarlyStopping
import myPlots

################################################################
# decide for how many subjects you want to compute the model at once.

# subjects = ["AMUSE_kw", "AMUSE_faz", "AMUSE_fce", "AMUSE_fcg", "AMUSE_fcj"]
subjects = ["AMUSE_kw"]

################################################################
# a0                    as it is
# n0 -> base for f.     subtract the mean of the 44 previous samples
# n1                    normalize
# f0                    n0 and butter
# f1                    n0 and cheby1               
# f2                    n0 and cheby2               
# f3                    n0 and ellip                
# f4                    n0 and firpm

select = "n0"           #@param {type:"string"}

################################################################
# to set for k-fold crossvalidation.
K_folds = 4                     #@param {type:"slider", min:2, max:8, step:1}
reproducibility_flag = False    #@param {type:"boolean"}
shuff = False                   #@param {type:"boolean"}
if reproducibility_flag:
    np.random.seed(42)
    set_seed(42)
    kfold = StratifiedKFold(n_splits=K_folds, shuffle=shuff, random_state=42)
else:
    kfold = StratifiedKFold(n_splits=K_folds, shuffle=shuff)

################################################################
b_optimizer_selected = 'Adam'   #@param ["Adam", "RMSprop", "SGD"]
b_learn_rate = 0.002            #@param {type:"number"}
b_momentum = 0.9                #@param {type:"number"}
b_rho = 0.9                     #@param {type:"number"}

b_n_epochs = 100                #@param {type:"slider", min:10, max:1000, step:10}
b_patience = 20                 #@param {type:"slider", min:5, max:1000, step:5}
b_class_weights = {0:1, 1:1}    #param {type:"raw"}

b_dropoutRate = 0.6                 #@param {type:"number"}
b_kernLength =  64                  #@param {type:"integer"}
b_F1 = 8                            #@param {type:"integer"}
b_D = 2                             #@param {type:"integer"}
b_F2 = b_F1 * b_D
b_norm_rate = 0.25                  #param {type:"number"}
b_dropoutType = 'SpatialDropout2D'  #param ["Dropout", "SpatialDropout2D"]

################################################################
m_optimizer_selected = 'Adam'   #@param ["Adam", "RMSprop", "SGD"]
m_learn_rate = 0.001            #@param {type:"number"}
m_momentum = 0.9                #@param {type:"number"}
m_rho = 0.9                     #@param {type:"number"}.

m_n_epochs = 200                                    #@param {type:"slider", min:10, max:1000, step:10}
m_patience = 30                                     #@param {type:"slider", min:5, max:1000, step:5}
m_class_weights = {0:1, 1:1, 2:1, 3:1, 4:1, 5:1}    #param {type:"raw"}

m_dropoutRate = 0.8                 #@param {type:"number"}
m_kernLength =  64                  #@param {type:"integer"}
m_F1 = 8                            #@param {type:"integer"}
m_D = 2                             #@param {type:"integer"}
m_F2 = m_F1 * m_D
m_norm_rate = 0.25                  #param {type:"number"}
m_dropoutType = 'SpatialDropout2D'  #param ["Dropout", "SpatialDropout2D"]

################################################################
# this piece of code just select the optimizer.
if b_optimizer_selected == 'Adam':
        b_opt = Adam(learning_rate = b_learn_rate)
elif b_optimizer_selected == 'RMSprop':
        b_opt = RMSprop(learning_rate = b_learn_rate, 
                        rho = b_rho, 
                        momentum = b_momentum,)
elif b_optimizer_selected == 'SGD':
        b_opt = SGD(learning_rate = b_learn_rate, 
                    momentum = b_momentum)

if m_optimizer_selected == 'Adam':
        m_opt = Adam(learning_rate = m_learn_rate)
elif m_optimizer_selected == 'RMSprop':
        m_opt = RMSprop(learning_rate = m_learn_rate, 
                        rho = m_rho, 
                        momentum = m_momentum,)
elif m_optimizer_selected == 'SGD':
        m_opt = SGD(learning_rate = m_learn_rate, 
                    momentum = m_momentum)


##################################################################################
##################################################################################

# FROM HERE, YOU DON'T NEED TO CHANGE ANYTHING IN THE CODE (you should check the
# path of each read and write though).

##################################################################################
##################################################################################
##################################################################################
##################################################################################
# update the run's number at each execution of the program 
# (at first, you need to create a txt file that contains any 
# string in the first line and a number in the second line).
from datetime import date
date = date.today()
with open(path0 + 'number_of_execution.txt', 'r') as f:
    script = f.readlines()
    if str(date) == script[0].strip():
        number_run = int(script[1]) + 1
    else:
        number_run = 1

# start the session and write in a file the hyperparameters.
with open(path0 + 'saveAccuracy.txt', 'a') as f:
    f.write("------------------ S T A R T ------------------")
    f.write("\nS E S S I O N   - " + select + '_' + str(date) + '_run-' + str(number_run))
    f.write('\nBINARY: ')
    if b_optimizer_selected == 'Adam':
        f.write('Adam(learning_rate = ' + str(b_learn_rate) + ')')
    elif b_optimizer_selected == 'RMSprop':
        f.write('RMSprop(learning_rate = ' + str(b_learn_rate) + ', rho = ' 
                + str(b_rho) + ', momentum = ' + str(b_momentum) + ')')
    elif b_optimizer_selected == 'SGD':
        f.write('SGD(learning_rate = ' + str(b_learn_rate) + ', momentum = ' 
                + str(b_momentum) + ')')
    f.write(' | n_epochs: ' + str(b_n_epochs) + ' | dropoutRate: ' + str(b_dropoutRate) + 
            ' | kernLength: ' + str(b_kernLength) + ' | F1: ' + str(b_F1) + 
            ' | D: ' + str(b_D) + ' | F2: ' + str(b_F2))
    f.write('\nMULTICLASS: ')
    if m_optimizer_selected == 'Adam':
        f.write('Adam(learning_rate = ' + str(m_learn_rate) + ')')
    elif m_optimizer_selected == 'RMSprop':
        f.write('RMSprop(learning_rate = ' + str(m_learn_rate) + ', rho = ' 
                + str(m_rho) + ', momentum = ' + str(m_momentum) + ')')
    elif m_optimizer_selected == 'SGD':
        f.write('SGD(learning_rate = ' + str(m_learn_rate) + ', momentum = ' 
                + str(m_momentum) + ')')
    f.write(' | n_epochs: ' + str(m_n_epochs) + ' | dropoutRate: ' + str(m_dropoutRate) + 
            ' | kernLength: ' + str(m_kernLength) + ' | F1: ' + str(m_F1) + 
            ' | D: ' + str(m_D) + ' | F2: ' + str(m_F2) + '\n\n')

# start the session individually for each subject and write in a file the hyperparameters 
# (from your path, this for loop will write a (or append to) txt file for each of the 
# subjects you selected).
for subject in subjects:
    with open(path0 + subject +'/' + subject + '_saveAccuracy.txt', 'a') as f:
        f.write("------------------ S T A R T ------------------")
        f.write("\nS E S S I O N   - " + subject + '_' + select + '_' + str(date) + '_run-' + str(number_run))
        f.write('\nBINARY: ')
        if b_optimizer_selected == 'Adam':
            f.write('Adam(learning_rate = ' + str(b_learn_rate) + ')')
        elif b_optimizer_selected == 'RMSprop':
            f.write('RMSprop(learning_rate = ' + str(b_learn_rate) + ', rho = ' 
                    + str(b_rho) + ', momentum = ' + str(b_momentum) + ')')
        elif b_optimizer_selected == 'SGD':
            f.write('SGD(learning_rate = ' + str(b_learn_rate) + ', momentum = ' 
                    + str(b_momentum) + ')')
        f.write(' | n_epochs: ' + str(b_n_epochs) + ' | dropoutRate: ' + str(b_dropoutRate) + 
                ' | kernLength: ' + str(b_kernLength) + ' | F1: ' + str(b_F1) + 
                ' | D: ' + str(b_D) + ' | F2: ' + str(b_F2))
        f.write('\nMULTICLASS: ')
        if m_optimizer_selected == 'Adam':
            f.write('Adam(learning_rate = ' + str(m_learn_rate) + ')')
        elif m_optimizer_selected == 'RMSprop':
            f.write('RMSprop(learning_rate = ' + str(m_learn_rate) + ', rho = ' 
                    + str(m_rho) + ', momentum = ' + str(m_momentum) + ')')
        elif m_optimizer_selected == 'SGD':
            f.write('SGD(learning_rate = ' + str(m_learn_rate) + ', momentum = ' 
                    + str(m_momentum) + ')')
        f.write(' | n_epochs: ' + str(m_n_epochs) + ' | dropoutRate: ' + str(m_dropoutRate) + 
                ' | kernLength: ' + str(m_kernLength) + ' | F1: ' + str(m_F1) + 
                ' | D: ' + str(m_D) + ' | F2: ' + str(m_F2) + '\n\n')

# hyperParam.mat ###############################################################
# use the info about dataset chose, date and execution_number to write the hyper
# parameters for this execution.
print('PATH: ' + path0 + '\nSave the hyperparameters in a .mat...')
sio.savemat('/hyperParam_' + select + '_' + str(date) + '_run-' + str(number_run) + '.mat', 
            {"b_optimizer":b_optimizer_selected, "b_learn_rate":b_learn_rate, 
             "b_momentum":b_momentum, "b_rho":b_rho, "b_n_epochs":b_n_epochs, 
             "b_dropoutRate":b_dropoutRate, "b_kernLength":b_kernLength, 
             "b_F1":b_F1, "b_D":b_D, "b_F2":b_F2, 
             "m_optimizer":m_optimizer_selected, "m_learn_rate":m_learn_rate, 
             "m_momentum":m_momentum, "b_rho":b_rho, "m_n_epochs":m_n_epochs, 
             "m_dropoutRate":m_dropoutRate, "m_kernLength":m_kernLength, 
             "m_F1":m_F1, "m_D":m_D, "m_F2":m_F2})


#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
# loop for each subject.
kkk = 0
for subject in subjects:
    kkk += 1
    Title = subject + '__' + select + '_' + str(date) + '_run-' + str(number_run)
    path1 = path0 + subject + '/models_' + Title + '/'

    # check if the folder exists. If not, create it. Watch out for this, it can mess up your PC
    if os.path.exists(path1):
        print("\nFOLDER ALREADY EXISTING AT:\n\t" + path1)
    else:
        print("\nNEW FOLDER CREATED AT:\n\t" + path1)
        os.makedirs(path1)
    print('\nSubject ' + str(kkk) + ' of ' + str(len(subjects)) + '\nTitle: ' + Title)

    # If you have run my MATLAB code without changing the variables names, this works!
    print('\nLoading dataset b...')
    b_path = path0 + subject +'/b'
    mat_contents = sio.loadmat(b_path + select + '_X3D.mat') # ba0_X3D.mat || 4320 samples for everyone
    b_X_train    = mat_contents['b_X3D_train']
    mat_contents = sio.loadmat(b_path + select + '_X3D_val.mat') # ba0_X3D_val.mat || 5940 samples for AMUSE_kw
    b_X_test     = mat_contents['b_X3D_val']
    mat_contents = sio.loadmat(b_path + '_Y.mat') # b_Y.mat || 4320 samples for everyone
    b_Y_train    = mat_contents['b_Y_train'][:, -1]
    mat_contents = sio.loadmat(b_path + '_Y_val.mat') # b_Y_val.mat || 5940 samples for AMUSE_kw
    b_Y_test     = mat_contents['b_Y_val'][:, -1]

    print('Loading dataset m...')
    m_path = path0 + subject +'/m'    
    mat_contents = sio.loadmat(m_path + select + '_X3D.mat') # ma0_X3D.mat || 720 samples for everyone
    m_X_train    = mat_contents['m_X3D_train']
    mat_contents = sio.loadmat(m_path + select + '_X3D_val.mat') # ma0_X3D_val.mat || 990 samples for AMUSE_kw
    m_X_test     = mat_contents['m_X3D_val']
    mat_contents = sio.loadmat(m_path + '_Y.mat') # m_Y.mat || 720 samples for everyone
    m_Y_train    = mat_contents['m_Y_train'][:, -1]
    mat_contents = sio.loadmat(m_path + '_Y_val.mat') # m_Y_val.mat || 990 samples for AMUSE_kw
    m_Y_test     = mat_contents['m_Y_val'][:, -1]
    mat_contents    = sio.loadmat(m_path + '_Y_sound.mat') # m_Y_sound.mat || 720 samples for everyone
    m_Y_sound_train = mat_contents['m_Y_sound_train'][:, -1]
    mat_contents    = sio.loadmat(m_path + '_Y_sound_val.mat') # m_Y_sound_val.mat || 990 samples for AMUSE_kw
    m_Y_sound_test  = mat_contents['m_Y_sound_val'][:, -1]

    b_sample_train  = b_X_train.shape[0]
    m_sample_train  = m_X_train.shape[0]
    b_sample_test   = b_X_test.shape[0]
    m_sample_test   = m_X_test.shape[0]
    b_C = b_X_train.shape[1]
    m_C = m_X_train.shape[1]
    b_T = b_X_train.shape[2]
    m_T = m_X_train.shape[2]

    print('Hot encoding...')
    b_Y_train_hot    = np_utils.to_categorical(b_Y_train)
    b_Y_test_hot     = np_utils.to_categorical(b_Y_test)
    m_Y_train_hot    = np_utils.to_categorical(m_Y_train-1)
    m_Y_test_hot     = np_utils.to_categorical(m_Y_test-1)
    m_Y_sound_train_hot    = np_utils.to_categorical(m_Y_sound_train-1)
    m_Y_sound_test_hot     = np_utils.to_categorical(m_Y_sound_test-1)
    b_X_train_hot    = b_X_train.reshape(b_sample_train, b_C, b_T, 1)
    b_X_test_hot     = b_X_test.reshape(b_sample_test, b_C, b_T, 1)
    m_X_train_hot    = m_X_train.reshape(m_sample_train, m_C, m_T, 1)
    m_X_test_hot     = m_X_test.reshape(m_sample_test, m_C, m_T, 1)

    #########################################################################################################
    #########################################################################################################
    print('\n################################################################################')
    print('################################################################################')
    print('BINARY MODEL (K-folds = ' + str(K_folds) + ' cross-validation):')
    b_cvscores = []
    b_n_epochs_actual = []
    n_kfold = 0
    for b_tra, b_val in kfold.split(b_X_train, b_Y_train):
        K.clear_session()
        
        n_kfold += 1

        # set the paths as you wish.
        path2 = '_' + Title + '_kfold' + str(n_kfold)
        b_path = '_b' + path2
        print('\n### Subject = ' + str(kkk) + ' of ' + str(len(subjects)) + 
              '\n### Kfold = ' + str(n_kfold) + ' of ' + str(K_folds) + 
              '\n###\nSET: ' + path1 + 'SWAG' + b_path + '\n\tBINARY: ' + Title)

        # call the binary model #############################################################################
        b_model = EEGNet(nb_classes = 2, Chans = b_C, Samples = b_T, 
                        dropoutRate = b_dropoutRate, kernLength = b_kernLength, 
                        F1 = b_F1, D = b_D, F2 = b_F2, norm_rate = b_norm_rate, 
                        dropoutType = b_dropoutType, activFunct = 'sigmoid')
        # compile the model.
        b_model.compile(loss = 'binary_crossentropy', optimizer = b_opt, metrics = ['accuracy'])
        b_numParams    = b_model.count_params()
        # set the checkpointer (as you can see below, I tried to use the model checkpointer, but I was
        # receiving an error, so I changed approach).
        b_callback = ReturnBestEarlyStopping(monitor = "val_loss", mode='min', patience = b_patience, 
                                   verbose = 1, restore_best_weights = True)
        # checkpoint_path = path1 + 'checkpoint' + b_path + '.h5'
        # b_checkpointer = ModelCheckpoint(filepath = checkpoint_path, monitor = "val_loss", 
        #                                  mode='min', verbose = 1, save_best_only = True)
        # fit the model.
        print('\nFit the binary model (n_kfold = ' + str(n_kfold) + ' of ' + str(K_folds) + '):')
        b_fittedModel  = b_model.fit(b_X_train_hot[b_tra], b_Y_train_hot[b_tra], batch_size = 15, epochs = b_n_epochs, 
                                    verbose = 2, validation_data = (b_X_train_hot[b_val], b_Y_train_hot[b_val]), 
                                    callbacks = [b_callback], class_weight = b_class_weights)
        b_n_epochs_actual.append(len(b_fittedModel.history['loss']))
        print('\nn_epochs = ' + str(len(b_fittedModel.history['loss'])))
        # evaluate the model on the validation set (1/4 of the train dataset).        
        # b_model.load_weights(checkpoint_path)
        b_scores = b_model.evaluate(b_X_train_hot[b_val], b_Y_train_hot[b_val], verbose=1)
        print('\nModel metrics:')
        print("%s: %.2f%%" % (b_model.metrics_names[1], b_scores[1]*100))
        b_cvscores.append(b_scores[1]*100)

        #########################################################################################################
        # I know, it's bad. but it works, so who cares...
        if n_kfold == 1:
            b_probs_1     = b_model.predict(b_X_test_hot)
        elif n_kfold == 2:
            b_probs_2     = b_model.predict(b_X_test_hot)
        elif n_kfold == 3:
            b_probs_3     = b_model.predict(b_X_test_hot)
        elif n_kfold == 4:
            b_probs_4     = b_model.predict(b_X_test_hot)
        elif n_kfold == 5:
            b_probs_5     = b_model.predict(b_X_test_hot)
        elif n_kfold == 6:
            b_probs_6     = b_model.predict(b_X_test_hot)
        elif n_kfold == 7:
            b_probs_7     = b_model.predict(b_X_test_hot)
        elif n_kfold == 8:
            b_probs_8     = b_model.predict(b_X_test_hot)

        # fittedModel.jpg ###################################################################################
        text = path1 + 'fittedModel' + b_path + '.jpg'
        myPlots.fitModel('binary', b_fittedModel.history, False, text)

        # fittedModelZoom.jpg ###############################################################################
        text = path1 + 'fittedModelZoom' + b_path + '.jpg'
        myPlots.fitModel('binary', b_fittedModel.history, True, text)

        # fittedModel.mat ###################################################################################
        sio.savemat(path1 + 'fittedModelMAT' + b_path + '.mat', 
                    {"b_fittedModel_loss":b_fittedModel.history['loss'], 
                     "b_fittedModel_loss_val":b_fittedModel.history['val_loss'], 
                     "b_fittedModel_acc":b_fittedModel.history['accuracy'],
                     "b_fittedModel_acc_val":b_fittedModel.history['val_accuracy']})
        
    # print the mean and the standard deviation of the cross-validation scores.
    print('\nMean and the STD of the binary cross-validation scores: ')
    print("%.2f%% (+/- %.2f%%)" % (np.mean(b_cvscores), np.std(b_cvscores)))

    # same code repeated for the multiclass dataset...
    #########################################################################################################
    #########################################################################################################
    print('\n################################################################################')
    print('################################################################################')
    print('MULTICLASS MODEL (K-folds = ' + str(K_folds) + ' cross-validation):')
    m_cvscores = []
    m_n_epochs_actual = []
    n_kfold = 0
    for m_tra, m_val in kfold.split(m_X_train, m_Y_train):
        K.clear_session()

        n_kfold += 1
        path2 = '_' + Title + '_kfold' + str(n_kfold)
        m_path = '_m' + path2
        print('\n### Subject = ' + str(kkk) + ' of ' + str(len(subjects)) + 
              '\n### Kfold = ' + str(n_kfold) + ' of ' + str(K_folds) + 
              '\n###\n###\nSET: ' + path1 + 'SWAG' + m_path + '\n\tMULTICLASS: ' + Title)

        # call the multiclass model #########################################################################
        m_model = EEGNet(nb_classes = 6, Chans = m_C, Samples = m_T, 
                        dropoutRate = m_dropoutRate, kernLength = m_kernLength, 
                        F1 = m_F1, D = m_D, F2 = m_F2, norm_rate = m_norm_rate, 
                        dropoutType = m_dropoutType, activFunct = 'softmax')
        # compile the model.
        m_model.compile(loss = 'categorical_crossentropy', optimizer = m_opt, metrics = ['accuracy'])
        m_numParams    = m_model.count_params()
        # set the checkpointer.
        m_callback = ReturnBestEarlyStopping(monitor = "val_loss", mode='min', patience = m_patience, 
                                   verbose = 1, restore_best_weights = True)
        # checkpoint_path = path1 + 'checkpoint' + b_path + '.h5'
        # m_checkpointer = ModelCheckpoint(filepath = checkpoint_path, monitor = "val_loss", 
        #                                  mode='min', verbose = 1, save_best_only = True)
        # fit the model.
        print('\nFit the multiclass model (n_kfold = ' + str(n_kfold) + ' of ' + str(K_folds) + '):')
        m_fittedModel  = m_model.fit(m_X_train_hot[m_tra], m_Y_train_hot[m_tra], batch_size = 15, epochs = m_n_epochs, 
                                        verbose = 2, validation_data = (m_X_train_hot[m_val], m_Y_train_hot[m_val]), 
                                        callbacks = [m_callback], class_weight = m_class_weights)
        m_n_epochs_actual.append(len(m_fittedModel.history['loss']))
        print('\nn_epochs = ' + str(len(m_fittedModel.history['loss'])))
        # evaluate the model on the validation set (1/4 of the training dataset).
        # m_model.load_weights(checkpoint_path)
        m_scores = m_model.evaluate(m_X_train_hot[m_val], m_Y_train_hot[m_val], verbose=1)
        print('\nModel metrics:')
        print("%s: %.2f%%" % (m_model.metrics_names[1], m_scores[1]*100))
        m_cvscores.append(m_scores[1]*100)

        #########################################################################################################
        if n_kfold == 1:
            m_probs_1     = m_model.predict(m_X_test_hot)
        elif n_kfold == 2:
            m_probs_2     = m_model.predict(m_X_test_hot)
        elif n_kfold == 3:
            m_probs_3     = m_model.predict(m_X_test_hot)
        elif n_kfold == 4:
            m_probs_4     = m_model.predict(m_X_test_hot)
        elif n_kfold == 5:
            m_probs_5     = m_model.predict(m_X_test_hot)
        elif n_kfold == 6:
            m_probs_6     = m_model.predict(m_X_test_hot)
        elif n_kfold == 7:
            m_probs_7     = m_model.predict(m_X_test_hot)
        elif n_kfold == 8:
            m_probs_8     = m_model.predict(m_X_test_hot)

        # fittedModel.jpg ###################################################################################
        text = path1 + 'fittedModel' + m_path + '.jpg'
        myPlots.fitModel('multiclass', m_fittedModel.history, False, text)

        # fittedModelZoom.jpg ###############################################################################
        text = path1 + 'fittedModelZoom' + m_path + '.jpg'
        myPlots.fitModel('multiclass', m_fittedModel.history, True, text)

        # fittedModel.mat ###################################################################################
        sio.savemat(path1 + 'fittedModelMAT' + m_path + '.mat', 
                        {"m_fittedModel_loss":m_fittedModel.history['loss'], 
                         "m_fittedModel_loss_val":m_fittedModel.history['val_loss'], 
                         "m_fittedModel_acc":m_fittedModel.history['accuracy'],
                         "m_fittedModel_acc_val":m_fittedModel.history['val_accuracy']})

    # print the mean and the standard deviation of the cross-validation scores.
    print('\nMean and the STD of the multiclass cross-validation scores: ')
    print("%.2f%% (+/- %.2f%%)" % (np.mean(m_cvscores), np.std(m_cvscores)))

    b_cvscores = np.array(b_cvscores)
    m_cvscores = np.array(m_cvscores)
    b_cv_mean  = np.mean(b_cvscores)
    b_cv_std   = np.std(b_cvscores)
    m_cv_mean  = np.mean(m_cvscores)
    m_cv_std   = np.std(m_cvscores)


    with open(path0 + 'saveAccuracy.txt', 'a') as f:
        f.write("CV-scores of K" + str(K_folds) + ": " + 
                "b_cvscores = " + str(np.around(b_cvscores, 2)) + 
                " | b_cv_mean = "    + str(np.around(np.mean(b_cvscores), 2)) + 
                " | b_cv_std = "  + str(np.around(np.std(b_cvscores), 2)) + '\n'
                "m_cvscores = " + str(np.around(m_cvscores, 2)) + 
                " | m_cv_mean = " + str(np.around(np.mean(m_cvscores), 2)) + 
                " | m_cv_std = "  + str(np.around(np.std(m_cvscores), 2)) + '\n' 
                "b_actual_epochs: " + str(b_n_epochs_actual) + '\n' + 
                "m_actual_epochs: " + str(m_n_epochs_actual) + '\n\n')
    
    with open(path0 + subject +'/' + subject + '_saveAccuracy.txt', 'a') as f:
        f.write("CV-scores of K" + str(K_folds) + ":\n" + 
                "b_cvscores = " + str(np.around(b_cvscores, 2)) + 
                " | b_cv_mean = "    + str(np.around(np.mean(b_cvscores), 2)) + 
                " | b_cv_std = "  + str(np.around(np.std(b_cvscores), 2)) + '\n'
                "m_cvscores = " + str(np.around(m_cvscores, 2)) + 
                " | m_cv_mean = " + str(np.around(np.mean(m_cvscores), 2)) + 
                " | m_cv_std = "  + str(np.around(np.std(m_cvscores), 2)) + '\n' 
                "b_actual_epochs: " + str(b_n_epochs_actual) + '\n' + 
                "m_actual_epochs: " + str(m_n_epochs_actual) + '\n\n')

    sio.savemat(path1 + 'cvscores_' + Title + '.mat', 
                    {"b_cvScores":b_cvscores, 
                     "m_cvScores":m_cvscores, 
                     "b_cv_mean":b_cv_mean, 
                     "b_cv_std":b_cv_std, 
                     "m_cv_mean":m_cv_mean, 
                     "m_cv_std":m_cv_std,
                     "b_n_epochs_actual":b_n_epochs_actual,
                     "m_n_epochs_actual":m_n_epochs_actual})

    #########################################################################################################
    #########################################################################################################



    #########################################################################################################
    #########################################################################################################



    #########################################################################################################
    #########################################################################################################


    print('\n\n\n\nL O A D - M O D E L S #########################################################')
    for n_kfold in range(K_folds):

        path1 = path0 + subject +'/models_' + Title + '/'
        path2 = '_' + Title + '_kfold' + str(n_kfold + 1)
        b_path = '_b' + path2
        m_path = '_m' + path2
        print('\n###\n###\nSET: ' + path1 + 'SWAG' + path2)

        #########################################################################################################
        # I know, bad coding. But it works so who cares...
        if n_kfold == 0:
            b_probs = b_probs_1
            b_preds_bin_old = b_probs_1.argmax(axis = -1)
            m_preds_multi   = m_probs_1.argmax(axis = -1)
        elif n_kfold == 1:
            b_probs = b_probs_2
            b_preds_bin_old = b_probs_2.argmax(axis = -1)
            m_preds_multi   = m_probs_2.argmax(axis = -1)
        elif n_kfold == 2:
            b_probs = b_probs_3
            b_preds_bin_old = b_probs_3.argmax(axis = -1)
            m_preds_multi   = m_probs_3.argmax(axis = -1)
        elif n_kfold == 3:
            b_probs = b_probs_4
            b_preds_bin_old = b_probs_4.argmax(axis = -1)
            m_preds_multi   = m_probs_4.argmax(axis = -1)
        elif n_kfold == 4:
            b_probs = b_probs_5
            b_preds_bin_old = b_probs_4.argmax(axis = -1)
            m_preds_multi   = m_probs_5.argmax(axis = -1)
        elif n_kfold == 5:
            b_probs = b_probs_6
            b_preds_bin_old = b_probs_6.argmax(axis = -1)
            m_preds_multi   = m_probs_6.argmax(axis = -1)
        elif n_kfold == 6:
            b_probs = b_probs_7
            b_preds_bin_old = b_probs_7.argmax(axis = -1)
            m_preds_multi   = m_probs_7.argmax(axis = -1)
        elif n_kfold == 7:
            b_probs = b_probs_8
            b_preds_bin_old = b_probs_8.argmax(axis = -1)
            m_preds_multi   = m_probs_8.argmax(axis = -1)

        b_preds_multi = np.zeros(m_sample_test, dtype = int)
        for k in range(m_sample_test):
            b_preds_multi[k] = b_probs[np.array(range(6)) + 6*k, 1].argmax()

        b_preds_bin = np.zeros(b_sample_test, dtype=int)
        for k in range(m_sample_test):
            b_preds_bin[b_preds_multi[k] + k*6] = 1
        
        m_preds_bin = np.zeros(b_sample_test, dtype=int)
        for k in range(m_sample_test):
            m_preds_bin[m_preds_multi[k] + k*6] = 1
        
        b_preds_multi = b_preds_multi + 1
        m_preds_multi = m_preds_multi + 1

        #########################################################################################################
        print('\nPlot the confusion matrix...')
        b_confusion_bin   = confusion_matrix(b_Y_test, b_preds_bin)
        b_confusion_multi = confusion_matrix(m_Y_test, b_preds_multi)
        m_confusion_bin   = confusion_matrix(b_Y_test, m_preds_bin)
        m_confusion_multi = confusion_matrix(m_Y_test, m_preds_multi)
        text = path1 + 'confusionMatrix' + path2 + '.jpg'
        
        myPlots.confusionMat(b_confusion_bin, b_confusion_multi, m_confusion_bin, m_confusion_multi, text)

        # accuracy&confusionMat.mat #########################################################################
        print('\nSave the accuracy variables and the confusion matrix on test data in a .mat...')
        b_acc_bin   = accuracy_score(b_Y_test, b_preds_bin)
        b_acc_multi = accuracy_score(m_Y_test, b_preds_multi)
        m_acc_bin   = accuracy_score(b_Y_test, m_preds_bin)
        m_acc_multi = accuracy_score(m_Y_test, m_preds_multi)
        sio.savemat(path1 + 'accuracy' + path2 + '.mat', 
                     {"b_acc_bin":b_acc_bin, "b_acc_multi":b_acc_multi, 
                      "b_confusion_bin":b_confusion_bin, "b_confusion_multi":b_confusion_multi, 
                      "m_acc_bin":m_acc_bin, "m_acc_multi":m_acc_multi, 
                      "m_confusion_bin":m_confusion_bin, "m_confusion_multi":m_confusion_multi})

        #####################################################################################################
        print('\nPrint accuracy in a txt file to compare the same options accross subjects...')
        with open(path0 + 'saveAccuracy.txt', 'a') as f:
            f.write(Title + ' | kfold = ' + str(n_kfold + 1))
            f.write("\n Binary-corrected acc: "  + str(round(b_acc_bin*100, 1)) + "%")
            f.write(" | Binary-multiclass acc: " + str(round(b_acc_multi*100, 1)) + "%")
            f.write(" | Multiclass-binary acc: " + str(round(m_acc_bin*100, 1)) + "%")
            f.write(" | Multiclass acc: "        + str(round(m_acc_multi*100, 1)) + "%")
            f.write("\n\n")

        #####################################################################################################
        print('\nPrint accuracy in a txt file to compare different options within each subject...')
        with open(path0 + subject + '/' + subject + '_saveAccuracy.txt', 'a') as f:
            f.write(Title + ' | kfold = ' + str(n_kfold + 1))
            f.write("\n Binary-corrected acc: "  + str(round(b_acc_bin*100, 1)) + "%")
            f.write(" | Binary-multiclass acc: " + str(round(b_acc_multi*100, 1)) + "%")
            f.write(" | Multiclass-binary acc: " + str(round(m_acc_bin*100, 1)) + "%")
            f.write(" | Multiclass acc: "        + str(round(m_acc_multi*100, 1)) + "%")
            f.write("\n\n")

        #####################################################################################################
        print('\nSave status of parameters in a txt file...')
        with open(path1 + 'info' + path2 + '.txt', 'w') as f:
            f.write('#######  ' + Title + '  ##  kfold = ' + str(n_kfold + 1) + '  #######')
            f.write('\n\nBINARY: ')
            if b_optimizer_selected == 'Adam':
                f.write('Adam(learning_rate = ' + str(b_learn_rate) + ')')
            elif b_optimizer_selected == 'RMSprop':
                f.write('RMSprop(learning_rate = ' + str(b_learn_rate) + ', rho = ' 
                        + str(b_rho) + ', momentum = ' + str(b_momentum) + ')')
            elif b_optimizer_selected == 'SGD':
                f.write('SGD(learning_rate = ' + str(b_learn_rate) + ', momentum = ' 
                        + str(b_momentum) + ')')
            f.write(' | n_epochs: '    + str(b_n_epochs))
            f.write(' | dropoutRate: ' + str(b_dropoutRate))
            f.write(' | kernLength: '  + str(b_kernLength))
            f.write(' | F1: '          + str(b_F1))
            f.write(' | D: '           + str(b_D))
            f.write(' | F2: '          + str(b_F2))
            f.write('\nBINARY DATASETS ########')
            f.write('\n X_train shape: ' + str(b_X_train.shape))
            f.write('\n X_test shape:  ' + str(b_X_test.shape))
            f.write('\n Y_train:       ' + str(b_Y_train.shape))
            f.write('\n Y_test :       ' + str(b_Y_test.shape))
            f.write('\n\nBINARY training & binary evaluation -------------------------\n')
            f.write('Accuracy = '           + str(round(b_acc_bin*100, 1)) + '%')
            f.write('\nConfusion matrix:\n' + str(b_confusion_bin))
            f.write('\n\nBINARY training & multiclass evaluation ---------------------\n')
            f.write('Accuracy = '           + str(round(b_acc_multi*100, 1)) + '%')
            f.write('\nConfusion matrix:\n' + str(b_confusion_multi))
            f.write('\n\nMULTICLASS: ')
            if m_optimizer_selected == 'Adam':
                f.write('Adam(learning_rate = ' + str(m_learn_rate) + ')')
            elif m_optimizer_selected == 'RMSprop':
                f.write('RMSprop(learning_rate = ' + str(m_learn_rate) + ', rho = ' 
                        + str(m_rho) + ', momentum = ' + str(m_momentum) + ')')
            elif m_optimizer_selected == 'SGD':
                f.write('SGD(learning_rate = ' + str(m_learn_rate) + ', momentum = ' 
                        + str(m_momentum) + ')')
            f.write(' | n_epochs: '    + str(m_n_epochs))
            f.write(' | dropoutRate: ' + str(m_dropoutRate))
            f.write(' | kernLength: '  + str(m_kernLength))
            f.write(' | F1: '          + str(m_F1))
            f.write(' | D: '           + str(m_D))
            f.write(' | F2: '          + str(m_F2))
            f.write('\MULTICLASS DATASETS #####')
            f.write('\n X_train shape: ' + str(b_X_train.shape))
            f.write('\n X_test shape:  ' + str(b_X_test.shape))
            f.write('\n Y_train:       ' + str(b_Y_train.shape))
            f.write('\n Y_test :       ' + str(b_Y_test.shape))
            f.write('\n\nMULTICLASS training & binary evaluation ----------------------\n')
            f.write('Accuracy = '           + str(round(m_acc_bin*100, 1)) + '%')
            f.write('\nConfusion matrix:\n' + str(m_confusion_bin))
            f.write('\n\nMULTICLASS training & multiclass evaluation ------------------\n')
            f.write('Accuracy = '           + str(round(m_acc_multi*100, 1)) + '%')
            f.write('\nConfusion matrix:\n' + str(m_confusion_multi))
            f.write("\n\nCV-scores of K" + str(K_folds) + ":\n" + 
                "b_cvscores = " + str(np.around(b_cvscores, 2)) + 
                " | b_cv_mean = "    + str(np.around(np.mean(b_cvscores), 2)) + 
                " | b_cv_std = "  + str(np.around(np.std(b_cvscores), 2)) + '\n'
                "m_cvscores = " + str(np.around(m_cvscores, 2)) + 
                " | m_cv_mean = " + str(np.around(np.mean(m_cvscores), 2)) + 
                " | m_cv_std = "  + str(np.around(np.std(m_cvscores), 2)) + '\n' 
                "b_actual_epochs: " + str(b_n_epochs_actual) + '\n' + 
                "m_actual_epochs: " + str(m_n_epochs_actual))
            
print("\n\n--------- E N D   O F   S E S S I O N ---------\n\n\n")
with open(path0 + 'saveAccuracy.txt', 'a') as f:
    f.write("--------- E N D   O F   S E S S I O N ---------\n\n\n")

with open(path0 + 'number_of_execution.txt', 'w') as f:
    f.write(str(date) + '\n' + str(number_run))
