####################################################################################
####################################################################################
####################################################################################
def main():
    import numpy as np
    import scipy.io as sio
    import os
    import sys
    sys.path.insert(0,'/content/drive/MyDrive/GoogleColab')

    from sklearn.metrics import confusion_matrix, accuracy_score
    from tensorflow.keras import utils as np_utils
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam, RMSprop, SGD
    from tensorflow.keras import backend as K
    K.set_image_data_format('channels_last')

    ################################################################################
    path0 = '/content/drive/MyDrive/GoogleColab/'
    import myCNN
    import myPlots
    import myFunc

    ################################################################
    ppp = '++++++++++++++++++++++++++++++++++++++++++++++++++++++++' #@param {type:"raw"}
    b_optimizer_selected = 'AMSgrad' #@param ["Adam", "AMSgrad", "RMSprop", "SGD"]
    b_learn_rate = 0.0005 #@param {type:"number"}
    b_param1 = 0.9 #@param {type:"number"}
    b_param2 = 0.99 #@param {type:"number"}

    b_n_epochs = 2000 #@param {type:"slider", min:10, max:3000, step:10}
    b_batch_size = 64 #@param {type:"slider", min:15, max:512, step:1}
    b_patience = 200 #@param {type:"slider", min:5, max:1000, step:5}
    b_class_weights = {0:1, 1:1} #param {type:"raw"}

    b_dropoutRate = 0.7 #@param {type:"number"}
    b_kernLength = 32 #@param {type:"integer"}
    b_F1 = 16 #@param {type:"integer"}
    b_D = 2 #@param {type:"integer"}
    b_F2 = 32
    b_kernLength2 = 16 #@param {type:"integer"}
    b_norm_rate = 0.25 #param {type:"number"}
    b_dropoutType = 'SpatialDropout2D' #param ["Dropout", "SpatialDropout2D"]

    ################################################################
    qqq = '++++++++++++++++++++++++++++++++++++++++++++++++++++++++' #@param {type:"raw"}
    m_optimizer_selected = 'AMSgrad' #@param ["Adam", "AMSgrad", "RMSprop", "SGD"]
    m_learn_rate = 0.0005 #@param {type:"number"}
    m_param1 = 0.9 #@param {type:"number"}
    m_param2 = 0.99 #@param {type:"number"}

    m_n_epochs = 10 # 4000 #@param {type:"slider", min:10, max:4000, step:10}
    m_batch_size = 64 #@param {type:"slider", min:15, max:512, step:1}
    m_patience = 10 # 300 #@param {type:"slider", min:5, max:1000, step:5}
    m_class_weights = {0:1, 1:1, 2:1, 3:1, 4:1, 5:1} #param {type:"raw"}

    m_dropoutRate = 0.75 #@param {type:"number"}
    m_kernLength = 32 #@param {type:"integer"}
    m_F1 = 16 #@param {type:"integer"}
    m_D = 2 #@param {type:"integer"}
    m_F2 = 32
    m_kernLength2 = 16 #@param {type:"integer"}
    m_norm_rate = 0.25 #param {type:"number"}
    m_dropoutType = 'SpatialDropout2D' #param ["Dropout", "SpatialDropout2D"]

    ################################################################ CHANGE CHANGE

    # selects = ["a0", "n0", "f0", "f1", "f2", "f3", "f4"]
    # selects = ["f03", "f04", "f05", "f07", "f10", "f15", "f20"]
    selects = ["f03", "f04", "f05", "f07", "f10", "f15"]
    # selects = ["f2"]

    # subjects = ["AMUSE_faz", "AMUSE_fce", "AMUSE_fcg", "AMUSE_fcj", "AMUSE_kw"]
    subjects = ["AMUSE_faz", "AMUSE_kw", "AMUSE_fcg"]

    dBdB = "3457" # ["8080", "3457"] filter's bandstop attenuation and ripple attenuation.
    bm_C = 45     # [45, 55] select how many channels you want to select.
    b_T = 256     # binary dataset's time samples of each sample and each channel.
    m_T = 128*3   # multiclass dataset's time samples of each sample and each channel.
    T_base = 44   # how many time samples taken from before the stimulus onset you want to average?
    T_shift = 0   # how many time samples you want to shift to better capture the ERP?

    ################################################################ CHANGE CHANGE CHANGE

    param_k = 0
    # quant1_label = [8, 16, 32]
    # for quant1 in quant1_label:
    #     b_F1 = quant1
    #     m_F1 = quant1

        # quant2_label = range(3)
        # for quant2 in quant2_label:
        #     b_F2 = quant2
        #     m_F2 = quant2
                    
            # for quant3 in quant3_label:
            #     b_kernLength2 = quant3
            #     m_kernLength2 = quant3

    quant4_label = range(2)
    for quant4 in quant4_label:
        param_k += 1

        sel_k = 0
        for select in selects:
            sel_k += 1

            ################################################################
            # just select the optimizer.
            b_opt = myFunc.SelectOptimizer(b_optimizer_selected, b_learn_rate, b_param1, b_param2)
            m_opt = myFunc.SelectOptimizer(m_optimizer_selected, m_learn_rate, m_param1, m_param2)

            ################################################################################
            ################################################################################
            ################################################################################
            # update the run's number at each execution of the program.
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
                f.write("\nS E S S I O N   -  " + str(date) + '  |  run = ' + str(number_run))
                f.write('\nBINARY:')
                if b_optimizer_selected == 'Adam':
                    f.write('\t\tAdam(learning_rate = ' + str(b_learn_rate) + ', beta_1 = ' 
                            + str(b_param1) + ', beta_2 = ' + str(b_param2) + ')')
                elif m_optimizer_selected == 'AMSgrad':
                    f.write('\t\tAMSgrad(learning_rate = ' + str(b_learn_rate) + ', beta_1 = ' 
                            + str(b_param1) + ', beta_2 = ' + str(b_param2) + ')')
                elif b_optimizer_selected == 'RMSprop':
                    f.write('\t\tRMSprop(learning_rate = ' + str(b_learn_rate) + ', rho = ' 
                            + str(b_param1) + ', momentum = ' + str(b_param2) + ')')
                elif b_optimizer_selected == 'SGD':
                    f.write('\t\tSGD(learning_rate = ' + str(b_learn_rate) + ', momentum = ' 
                            + str(b_param2) + ')')
                f.write(' | n_epochs: ' + str(b_n_epochs) + ' | patience: ' + str(b_patience) + 
                        ' | batch_size: ' + str(b_batch_size) + ' | dropoutRate: ' + str(b_dropoutRate) + 
                        '\n\t\tkernLength: ' + str(b_kernLength) + ' | F1: ' + str(b_F1) + ' | D: ' + 
                        str(b_D) + ' | F2: ' + str(b_F2) + ' | kernLength2: ' + str(b_kernLength2))
                f.write('\nMULTICLASS:')
                if m_optimizer_selected == 'Adam':
                    f.write('\tAdam(learning_rate = ' + str(m_learn_rate) + ', beta_1 = ' 
                            + str(m_param1) + ', beta_2 = ' + str(m_param2) + ')')
                elif m_optimizer_selected == 'AMSgrad':
                    f.write('\tAMSgrad(learning_rate = ' + str(m_learn_rate) + ', beta_1 = ' 
                            + str(m_param1) + ', beta_2 = ' + str(m_param2) + ')')
                elif m_optimizer_selected == 'RMSprop':
                    f.write('\tRMSprop(learning_rate = ' + str(m_learn_rate) + ', rho = ' 
                            + str(m_param1) + ', momentum = ' + str(m_param2) + ')')
                elif m_optimizer_selected == 'SGD':
                    f.write('\tSGD(learning_rate = ' + str(m_learn_rate) + ', momentum = ' 
                            + str(m_param2) + ')')
                f.write(' | n_epochs: ' + str(m_n_epochs) + ' | patience: ' + str(m_patience) + 
                        ' | batch_size: ' + str(m_batch_size) + ' | dropoutRate: ' + str(m_dropoutRate) + 
                        '\n\t\tkernLength: ' + str(m_kernLength) + ' | F1: ' + str(m_F1) + ' | D: ' + 
                        str(m_D) + ' | F2: ' + str(m_F2) + ' | kernLength2: ' + str(m_kernLength2) + '\n')

            # start the session individually for each subject and write in a file the hyperparameters.
            for subject in subjects:
                with open(path0 + subject +'/' + subject + '_saveAccuracy.txt', 'a') as f:
                    f.write("------------------ S T A R T ------------------")
                    f.write("\nS E S S I O N   -  " + subject + '  |  ' + str(date) + '  |  run = ' + 
                            str(number_run))
                    f.write('\nBINARY:')
                    if b_optimizer_selected == 'Adam':
                        f.write('\t\tAdam(learning_rate = ' + str(b_learn_rate) + ', beta_1 = ' 
                                + str(b_param1) + ', beta_2 = ' + str(b_param2) + ')')
                    elif m_optimizer_selected == 'AMSgrad':
                        f.write('\t\tAMSgrad(learning_rate = ' + str(b_learn_rate) + ', beta_1 = ' 
                                + str(b_param1) + ', beta_2 = ' + str(b_param2) + ')')
                    elif b_optimizer_selected == 'RMSprop':
                        f.write('\t\tRMSprop(learning_rate = ' + str(b_learn_rate) + ', rho = ' 
                                + str(b_param1) + ', momentum = ' + str(b_param2) + ')')
                    elif b_optimizer_selected == 'SGD':
                        f.write('\t\tSGD(learning_rate = ' + str(b_learn_rate) + ', momentum = ' 
                                + str(b_param2) + ')')
                    f.write(' | n_epochs: ' + str(b_n_epochs) + ' | patience: ' + str(b_patience) + 
                            ' | batch_size: ' + str(b_batch_size) + ' | dropoutRate: ' + str(b_dropoutRate) + 
                            '\n\t\tkernLength: ' + str(b_kernLength) + ' | F1: ' + str(b_F1) + ' | D: ' + 
                            str(b_D) + ' | F2: ' + str(b_F2) + ' | kernLength2: ' + str(b_kernLength2))
                    f.write('\nMULTICLASS:')
                    if m_optimizer_selected == 'Adam':
                        f.write('\tAdam(learning_rate = ' + str(m_learn_rate) + ', beta_1 = ' 
                                + str(m_param1) + ', beta_2 = ' + str(m_param2) + ')')
                    elif m_optimizer_selected == 'AMSgrad':
                        f.write('\tAMSgrad(learning_rate = ' + str(m_learn_rate) + ', beta_1 = ' 
                                + str(m_param1) + ', beta_2 = ' + str(m_param2) + ')')
                    elif m_optimizer_selected == 'RMSprop':
                        f.write('\tRMSprop(learning_rate = ' + str(m_learn_rate) + ', rho = ' 
                                + str(m_param1) + ', momentum = ' + str(m_param2) + ')')
                    elif m_optimizer_selected == 'SGD':
                        f.write('\tSGD(learning_rate = ' + str(m_learn_rate) + ', momentum = ' 
                                + str(m_param2) + ')')
                    f.write(' | n_epochs: ' + str(m_n_epochs) + ' | patience: ' + str(m_patience) + 
                            ' | batch_size: ' + str(m_batch_size) + ' | dropoutRate: ' + str(m_dropoutRate) + 
                            '\n\t\tkernLength: ' + str(m_kernLength) + ' | F1: ' + str(m_F1) + ' | D: ' + 
                            str(m_D) + ' | F2: ' + str(m_F2) + ' | kernLength2: ' + str(m_kernLength2) + '\n')
            # hyperParam.mat ###############################################################
            print('PATH: ' + path0 + '\nSave the hyperparameters in a .mat...')
            sio.savemat('/hyperParam_' + select + '_' + str(date) + '_run-' + str(number_run) + '.mat', 
                        {"b_optimizer":b_optimizer_selected,       "b_learn_rate":b_learn_rate, 
                        "b_param1":b_param1, "b_param2":b_param2, "b_batch_size":b_batch_size, 
                        "b_n_epochs":b_n_epochs,                  "b_patience":b_patience, 
                        "b_dropoutRate":b_dropoutRate,            "b_kernLength":b_kernLength, 
                        "b_F1":b_F1, "b_D":b_D, "b_F2":b_F2,      "b_kernLength2":b_kernLength2, 
                        "m_optimizer":m_optimizer_selected,       "m_learn_rate":m_learn_rate, 
                        "m_param1":m_param1, "m_param2":m_param2, "m_batch_size":m_batch_size, 
                        "m_n_epochs":m_n_epochs,                  "m_patience":m_patience, 
                        "m_dropoutRate":m_dropoutRate,            "m_kernLength":m_kernLength, 
                        "m_F1":m_F1, "m_D":m_D, "m_F2":m_F2,      "m_kernLength2":m_kernLength2})
                    
            
            ################################################################################
            ################################################################################
            ################################################################################

            ################################################################################
            ################################################################################
            ################################################################################

            ################################################################################
            ################################################################################
            # loop for each subject.
            sub_k = 0
            for subject in subjects:
                sub_k += 1
                
                Title = subject + '_' + str(date) + '_run-' + str(number_run) + '__' + select
                path1 = path0 + subject + '/models_' + Title + '/'
                if os.path.exists(path1):
                    print("\nFOLDER ALREADY EXISTING AT:\n\t" + path1)
                else:
                    print("\nNEW FOLDER CREATED AT:\n\t" + path1)
                    os.makedirs(path1)
                print('\nSubject ' + str(sub_k) + ' of ' + str(len(subjects)) + '\nTitle: ' + Title)
                
                ############################################################################
                print("------------------ S T A R T ------------------")
                print("S E S S I O N   -  " + subject + '  |  ' + str(date) + '  |  run = ' + str(number_run))
                print('BINARY:')
                if b_optimizer_selected == 'Adam':
                    print('Adam(learning_rate = ' + str(b_learn_rate) + ', beta_1 = ' 
                            + str(b_param1) + ', beta_2 = ' + str(b_param2) + ')')
                if b_optimizer_selected == 'AMSgrad':
                    print('AMSgrad(learning_rate = ' + str(b_learn_rate) + ', beta_1 = ' 
                            + str(b_param1) + ', beta_2 = ' + str(b_param2) + ')')
                elif b_optimizer_selected == 'RMSprop':
                    print('RMSprop(learning_rate = ' + str(b_learn_rate) + ', rho = ' 
                            + str(b_param1) + ', momentum = ' + str(b_param2) + ')')
                elif b_optimizer_selected == 'SGD':
                    print('SGD(learning_rate = ' + str(b_learn_rate) + ', momentum = ' 
                            + str(b_param2) + ')')
                print('#\tn_epochs: ' + str(b_n_epochs) + ' | patience: ' + str(b_patience) + 
                    ' | batch_size: ' + str(b_batch_size) + ' | dropoutRate: ' + str(b_dropoutRate) + 
                    '\n#\tkernLength: ' + str(b_kernLength) + ' | F1: ' + str(b_F1) + ' | D: ' + 
                    str(b_D) + ' | F2: ' + str(b_F2) + ' | kernLength2: ' + str(b_kernLength2))
                print('MULTICLASS:')
                if m_optimizer_selected == 'Adam':
                    print('Adam(learning_rate = ' + str(m_learn_rate) + ', beta_1 = ' 
                            + str(m_param1) + ', beta_2 = ' + str(m_param2) + ')')
                elif m_optimizer_selected == 'AMSgrad':
                    print('AMSgrad(learning_rate = ' + str(m_learn_rate) + ', beta_1 = ' 
                            + str(m_param1) + ', beta_2 = ' + str(m_param2) + ')')
                elif m_optimizer_selected == 'RMSprop':
                    print('RMSprop(learning_rate = ' + str(m_learn_rate) + ', rho = ' 
                            + str(m_param1) + ', momentum = ' + str(m_param2) + ')')
                elif m_optimizer_selected == 'SGD':
                    print('SGD(learning_rate = ' + str(m_learn_rate) + ', momentum = ' 
                            + str(m_param2) + ')')
                print('#\tn_epochs: ' + str(m_n_epochs) + ' | patience: ' + str(m_patience) + 
                    ' | batch_size: ' + str(m_batch_size) + ' | dropoutRate: ' + str(m_dropoutRate) + 
                    '\n#\tkernLength: ' + str(m_kernLength) + ' | F1: ' + str(m_F1) + ' | D: ' + 
                    str(m_D) + ' | F2: ' + str(m_F2) + ' | kernLength2: ' + str(m_kernLength2) + '\n')

                ############################################################################
                print('\nLoading datasets...')
                b_X_train, b_X_test, m_X_train, m_X_test = myFunc.LoadDataset(subject, 
                                        select, dBdB, bm_C, b_T, m_T, T_base, T_shift, path0)

                ############################################################################
                # print('\nLoading dataset b...') ############################################
                b_path = path0 + subject +'/b'
                # mat_contents = sio.loadmat(b_path + select + '_X3D.mat') # ba0_X3D.mat || 4320 samples
                # b_X_train    = mat_contents['b_X3D_train'] #################################
                # mat_contents = sio.loadmat(b_path + select + '_X3D_val.mat') # ba0_X3D_val.mat || 5940 samples
                # b_X_test     = mat_contents['b_X3D_val'] ###################################
                mat_contents = sio.loadmat(b_path + '_Y.mat') # b_Y.mat || 4320 samples
                b_Y_train    = mat_contents['b_Y_train'][:, -1]
                mat_contents = sio.loadmat(b_path + '_Y_val.mat') # b_Y_val.mat || 5940 samples
                b_Y_test     = mat_contents['b_Y_val'][:, -1]

                ############################################################################
                # print('Loading dataset m...') ##############################################
                m_path = path0 + subject +'/m'    
                # mat_contents = sio.loadmat(m_path + select + '_X3D.mat') # ma0_X3D.mat || 720 samples
                # m_X_train    = mat_contents['m_X3D_train'] #################################
                # mat_contents = sio.loadmat(m_path + select + '_X3D_val.mat') # ma0_X3D_val.mat || 990 samples
                # m_X_test     = mat_contents['m_X3D_val'] ###################################
                mat_contents = sio.loadmat(m_path + '_Y.mat') # m_Y.mat || 720 samples
                m_Y_train    = mat_contents['m_Y_train'][:, -1]
                mat_contents = sio.loadmat(m_path + '_Y_val.mat') # m_Y_val.mat || 990 samples
                m_Y_test     = mat_contents['m_Y_val'][:, -1]
                mat_contents    = sio.loadmat(m_path + '_Y_sound.mat') # m_Y_sound.mat || 720 samples
                m_Y_sound_train = mat_contents['m_Y_sound_train'][:, -1]
                mat_contents    = sio.loadmat(m_path + '_Y_sound_val.mat') # m_Y_sound_val.mat || 990 samples
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

                #########################################################################################################
                #########################################################################################################


                print('\n################################################################################')
                print('################################################################################')
                print('BINARY MODEL:')
                if quant4 == 0:
                    b_tra = np.array(range(int(b_sample_train*3/4)))
                    b_val = np.array(range(int(b_sample_train/4))) + int(b_sample_train*3/4)
                else:
                    b_val = np.array(range(int(b_sample_train/4)))
                    b_tra = np.array(range(int(b_sample_train*3/4))) + int(b_sample_train/4)
                K.clear_session()
                
                path2 = '_' + Title
                b_path = '_b' + path2
                print('\n### PARAM = ' + str(param_k) + ', SELECT = '  + str(sel_k) + 
                    '\n###\n### Subject = ' + str(sub_k) + ' of ' + str(len(subjects)) + 
                    '\n###\n###\nSET: ' + path1 + 'SWAG' + b_path + '\n\tBINARY: ' + Title)

                # call the binary model #############################################################################
                b_model = myCNN.EEGNet(nb_classes = 2, Chans = b_C, Samples = b_T, 
                                    dropoutRate = b_dropoutRate, kernLength = b_kernLength, 
                                    F1 = b_F1, D = b_D, F2 = b_F2, kernLength2 = b_kernLength2, 
                                    norm_rate = b_norm_rate, dropoutType = b_dropoutType, 
                                    activFunct = 'sigmoid')
                # compile the model.
                b_model.compile(loss = 'binary_crossentropy', optimizer = b_opt, metrics = ['accuracy'])
                b_numParams = b_model.count_params()

                if sub_k * sel_k == 1:
                    # save the model's weights only at the the first execution.
                    b_model.save_weights(path0 + 'b_model_weights.h5')
                else:
                    b_model.load_weights(path0 + 'b_model_weights.h5')

                # set the checkpointer.
                b_callback = myCNN.ReturnBestEarlyStopping(monitor = "val_loss", mode='min', patience = b_patience, 
                                                        verbose = 1, restore_best_weights = True)
                # fit the model.
                print('\nFit the binary model:')
                b_fittedModel = b_model.fit(b_X_train_hot[b_tra], b_Y_train_hot[b_tra], 
                                            batch_size = b_batch_size, epochs = b_n_epochs, 
                                            verbose = 2, validation_data = (b_X_train_hot[b_val], 
                                            b_Y_train_hot[b_val]), callbacks = [b_callback], 
                                            class_weight = b_class_weights)
                b_n_epochs_actual = len(b_fittedModel.history['loss'])
                print('\nn_epochs = ' + str(b_n_epochs_actual))

                # evaluate the model on the validation set (1/4 of the train dataset).        
                b_scores = b_model.evaluate(b_X_train_hot[b_val], b_Y_train_hot[b_val], verbose=1)
                print('\nModel metrics:')
                print("%s: %.2f%%" % (b_model.metrics_names[1], b_scores[1]*100))

                #####################################################################################################
                b_probs = b_model.predict(b_X_test_hot)

                # fittedModel.jpg ###################################################################################
                text = path1 + 'fittedModel' + b_path + '.jpg'
                myPlots.fitModel('binary', b_fittedModel.history, 0, text)

                # fittedModelZoom.jpg ###############################################################################
                text = path1 + 'fittedModelZoom' + b_path + '.jpg'
                myPlots.fitModel('binary', b_fittedModel.history, 1, text)

                # fittedModelScale.jpg ###############################################################################
                text = path1 + 'fittedModelScale' + b_path + '.jpg'
                myPlots.fitModel('binary', b_fittedModel.history, 2, text)

                # fittedModel.mat ###################################################################################
                sio.savemat(path1 + 'fittedModelMAT' + b_path + '.mat', 
                            {"b_fittedModel_loss":b_fittedModel.history['loss'], 
                            "b_fittedModel_loss_val":b_fittedModel.history['val_loss'], 
                            "b_fittedModel_acc":b_fittedModel.history['accuracy'],
                            "b_fittedModel_acc_val":b_fittedModel.history['val_accuracy']})
                

                #########################################################################################################
                #########################################################################################################

                #########################################################################################################
                #########################################################################################################


                print('\n################################################################################')
                print('################################################################################')
                print('MULTICLASS MODEL:')
                if quant4 == 0:
                    m_tra = np.array(range(int(m_sample_train*3/4)))
                    m_val = np.array(range(int(m_sample_train/4))) + int(m_sample_train*3/4)
                else:
                    m_val = np.array(range(int(m_sample_train/4)))
                    m_tra = np.array(range(int(m_sample_train*3/4))) + int(m_sample_train/4)
                K.clear_session()

                path2 = '_' + Title
                m_path = '_m' + path2
                print('\n### PARAM = ' + str(param_k) + ', SELECT = '  + str(sel_k) + 
                    '\n###\n### Subject = ' + str(sub_k) + ' of ' + str(len(subjects)) + 
                    '\n###\n###\nSET: ' + path1 + 'SWAG' + m_path + '\n\tMULTICLASS: ' + Title)

                # call the multiclass model #########################################################################
                m_model = myCNN.EEGNet(nb_classes = 6, Chans = m_C, Samples = m_T, 
                                    dropoutRate = m_dropoutRate, kernLength = m_kernLength, 
                                    F1 = m_F1, D = m_D, F2 = m_F2, kernLength2 = m_kernLength2, 
                                    norm_rate = m_norm_rate, dropoutType = m_dropoutType, 
                                    activFunct = 'softmax')
                # compile the model.
                m_model.compile(loss = 'categorical_crossentropy', optimizer = m_opt, metrics = ['accuracy'])
                m_numParams = m_model.count_params()

                if sub_k * sel_k == 1:
                    # save the model's weights only at the the first execution.
                    m_model.save_weights(path0 + 'm_model_weights.h5')
                else:
                    m_model.load_weights(path0 + 'm_model_weights.h5')
                    
                # set the checkpointer.
                m_callback = myCNN.ReturnBestEarlyStopping(monitor = "val_loss", mode='min', patience = m_patience, 
                                                        verbose = 1, restore_best_weights = True)
                # fit the model.
                print('\nFit the multiclass model:')
                m_fittedModel = m_model.fit(m_X_train_hot[m_tra], m_Y_train_hot[m_tra], 
                                            batch_size = m_batch_size, epochs = m_n_epochs, 
                                            verbose = 2, validation_data = (m_X_train_hot[m_val], 
                                            m_Y_train_hot[m_val]), callbacks = [m_callback], 
                                            class_weight = m_class_weights)
                m_n_epochs_actual = len(m_fittedModel.history['loss'])
                print('\nn_epochs = ' + str(m_n_epochs_actual))

                # evaluate the model on the validation set (1/4 of the training dataset).
                m_scores = m_model.evaluate(m_X_train_hot[m_val], m_Y_train_hot[m_val], verbose=1)
                print('\nModel metrics:')
                print("%s: %.2f%%" % (m_model.metrics_names[1], m_scores[1]*100))

                #########################################################################################################
                m_probs = m_model.predict(m_X_test_hot)

                # fittedModel.jpg ###################################################################################
                text = path1 + 'fittedModel' + m_path + '.jpg'
                myPlots.fitModel('multiclass', m_fittedModel.history, 0, text)

                # fittedModelZoom.jpg ###############################################################################
                text = path1 + 'fittedModelZoom' + m_path + '.jpg'
                myPlots.fitModel('multiclass', m_fittedModel.history, 1, text)

                # fittedModelScale.jpg ###############################################################################
                text = path1 + 'fittedModelScale' + m_path + '.jpg'
                myPlots.fitModel('multiclass', m_fittedModel.history, 2, text)

                # fittedModel.mat ###################################################################################
                sio.savemat(path1 + 'fittedModelMAT' + m_path + '.mat', 
                            {"m_fittedModel_loss":m_fittedModel.history['loss'], 
                            "m_fittedModel_loss_val":m_fittedModel.history['val_loss'], 
                            "m_fittedModel_acc":m_fittedModel.history['accuracy'],
                            "m_fittedModel_acc_val":m_fittedModel.history['val_accuracy']})


                #########################################################################################################
                #########################################################################################################



                #########################################################################################################
                #########################################################################################################



                #########################################################################################################
                #########################################################################################################


                print('\n\n\nL O A D - M O D E L S #########################################################')
                path1 = path0 + subject +'/models_' + Title + '/'
                path2 = '_' + Title
                b_path = '_b' + path2
                m_path = '_m' + path2
                print('\n###\n###\nSET: ' + path1 + 'SWAG' + path2)

                #########################################################################################################
                b_preds_bin_old = b_probs.argmax(axis = -1)
                m_preds_multi   = m_probs.argmax(axis = -1)

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
                            {"b_acc_val":b_scores[1],               "m_acc_val":m_scores[1], 
                            "b_probs":b_probs,                     "m_probs":m_probs, 
                            "b_acc_bin":b_acc_bin,                 "m_acc_bin":m_acc_bin, 
                            "b_acc_multi":b_acc_multi,             "m_acc_multi":m_acc_multi, 
                            "b_confusion_bin":b_confusion_bin,     "m_confusion_bin":m_confusion_bin, 
                            "b_confusion_multi":b_confusion_multi, "m_confusion_multi":m_confusion_multi, 
                            "b_n_epochs_actual":b_n_epochs_actual, "m_n_epochs_actual":m_n_epochs_actual})

                #####################################################################################################
                print('\nPrint accuracy in a txt file to compare the same options accross subjects...')
                with open(path0 + 'saveAccuracy.txt', 'a') as f:
                    f.write(subject + " | " + select)
                    f.write("\nBinary-corrected acc: "   + str(round(b_acc_bin*100, 1)) + "%")
                    f.write(" | Binary-multiclass acc: " + str(round(b_acc_multi*100, 1)) + "%")
                    f.write(" | Multiclass-binary acc: " + str(round(m_acc_bin*100, 1)) + "%")
                    f.write(" | Multiclass acc: "        + str(round(m_acc_multi*100, 1)) + "%")
                    f.write("\nBinary Validation acc: "                 + str(np.around(b_scores[1]*100, 1)) + "%")
                    f.write("\t\t\t\t\t\t| Multiclass Validation acc: " + str(np.around(m_scores[1]*100, 1)) + "%")
                    f.write("\nActual Epochs (bin): "    + str(b_n_epochs_actual) + "/" + str(b_n_epochs))
                    f.write(" | Actual Epochs (multi): " + str(m_n_epochs_actual) + "/" + str(m_n_epochs))
                    f.write("\n\n")

                #####################################################################################################
                print('\nPrint accuracy in a txt file to compare different options within each subject...')
                with open(path0 + subject + '/' + subject + '_saveAccuracy.txt', 'a') as f:
                    f.write(subject + " | " + select)
                    f.write("\nBinary-corrected acc: "  + str(round(b_acc_bin*100, 1)) + "%")
                    f.write(" | Binary-multiclass acc: " + str(round(b_acc_multi*100, 1)) + "%")
                    f.write(" | Multiclass-binary acc: " + str(round(m_acc_bin*100, 1)) + "%")
                    f.write(" | Multiclass acc: "        + str(round(m_acc_multi*100, 1)) + "%")
                    f.write("\nBinary Validation acc: "    + str(np.around(b_scores[1]*100, 1)) + "%")
                    f.write("\t\t\t\t\t\t| Multiclass Validation acc: " + str(np.around(m_scores[1]*100, 1)) + "%")
                    f.write("\nActual Epochs (bin): "    + str(b_n_epochs_actual) + "/" + str(b_n_epochs))
                    f.write(" | Actual Epochs (multi): " + str(m_n_epochs_actual) + "/" + str(m_n_epochs))
                    f.write("\n\n")

                #####################################################################################################
                print('\nSave status of parameters in a txt file...')
                with open(path1 + 'info' + path2 + '.txt', 'w') as f:
                    f.write('#######  ' + Title + '  #######')
                    f.write('\n\nBINARY: ')
                    if b_optimizer_selected == 'Adam':
                        f.write('\t\tAdam(learning_rate = ' + str(b_learn_rate) + ', beta_1 = ' 
                                + str(b_param1) + ', beta_2 = ' + str(b_param2) + ')')
                    elif b_optimizer_selected == 'AMSgrad':
                        f.write('\t\tAMSgrad(learning_rate = ' + str(b_learn_rate) + ', beta_1 = ' 
                                + str(b_param1) + ', beta_2 = ' + str(b_param2) + ')')
                    elif b_optimizer_selected == 'RMSprop':
                        f.write('\t\tRMSprop(learning_rate = ' + str(b_learn_rate) + ', rho = ' 
                                + str(b_param1) + ', momentum = ' + str(b_param2) + ')')
                    elif b_optimizer_selected == 'SGD':
                        f.write('\t\tSGD(learning_rate = ' + str(b_learn_rate) + ', momentum = ' 
                                + str(b_param2) + ')')
                    f.write(' | n_epochs: ' + str(b_n_epochs_actual) + "/" + str(b_n_epochs) + 
                            ' | patience: ' + str(b_patience) + ' | batch_size: ' + str(b_batch_size) + 
                            ' | dropoutRate: ' + str(b_dropoutRate) + 
                            '\n\t\tkernLength: ' + str(b_kernLength) + ' | F1: ' + str(b_F1) + ' | D: ' + 
                            str(b_D) + ' | F2: ' + str(b_F2) + ' | kernLength2: ' + str(b_kernLength2))
                    f.write('\nBINARY DATASETS ########')
                    f.write('\n X_train shape: ' + str(b_X_train.shape))
                    f.write('\n X_test shape:  ' + str(b_X_test.shape))
                    f.write('\n Y_train:       ' + str(b_Y_train.shape))
                    f.write('\n Y_test :       ' + str(b_Y_test.shape))
                    f.write('\n\nBINARY training & binary evaluation -------------------------\n')
                    f.write('Val score = '          + str(np.around(b_scores[1]*100, 1)) + '%')
                    f.write('\nTest accuracy = '    + str(round(b_acc_bin*100, 1)) + '%')
                    f.write('\nConfusion matrix:\n' + str(b_confusion_bin))
                    f.write('\n\nBINARY training & multiclass evaluation ---------------------\n')
                    f.write('Test accuracy = '      + str(round(b_acc_multi*100, 1)) + '%')
                    f.write('\nConfusion matrix:\n' + str(b_confusion_multi))
                    f.write('\n\nMULTICLASS: ')
                    if m_optimizer_selected == 'Adam':
                        f.write('\tAdam(learning_rate = ' + str(m_learn_rate) + ', beta_1 = ' 
                                + str(m_param1) + ', beta_2 = ' + str(m_param2) + ')')
                    elif m_optimizer_selected == 'AMSgrad':
                        f.write('\tAMSgrad(learning_rate = ' + str(m_learn_rate) + ', beta_1 = ' 
                                + str(m_param1) + ', beta_2 = ' + str(m_param2) + ')')
                    elif m_optimizer_selected == 'RMSprop':
                        f.write('\tRMSprop(learning_rate = ' + str(m_learn_rate) + ', rho = ' 
                                + str(m_param1) + ', momentum = ' + str(m_param2) + ')')
                    elif m_optimizer_selected == 'SGD':
                        f.write('\tSGD(learning_rate = ' + str(m_learn_rate) + ', momentum = ' 
                                + str(m_param2) + ')')
                    f.write(' | n_epochs: ' + str(m_n_epochs_actual) + "/" + str(m_n_epochs) + 
                            ' | patience: ' + str(m_patience) + ' | batch_size: ' + str(m_batch_size) + 
                            ' | dropoutRate: ' + str(m_dropoutRate) + 
                            '\n\t\tkernLength: ' + str(m_kernLength) + ' | F1: ' + str(m_F1) + ' | D: ' + 
                            str(m_D) + ' | F2: ' + str(m_F2) + ' | kernLength2: ' + str(m_kernLength2))
                    f.write('\nMULTICLASS DATASETS #####')
                    f.write('\n X_train shape: ' + str(m_X_train.shape))
                    f.write('\n X_test shape:  ' + str(m_X_test.shape))
                    f.write('\n Y_train:       ' + str(m_Y_train.shape))
                    f.write('\n Y_test :       ' + str(m_Y_test.shape))
                    f.write('\n\nMULTICLASS training & binary evaluation ----------------------\n')
                    f.write('Test accuracy = '      + str(round(m_acc_bin*100, 1)) + '%')
                    f.write('\nConfusion matrix:\n' + str(m_confusion_bin))
                    f.write('\n\nMULTICLASS training & multiclass evaluation ------------------\n')
                    f.write('Val score = '          + str(np.around(m_scores[1]*100, 1)) + '%')
                    f.write('\nTest accuracy = '    + str(round(m_acc_multi*100, 1)) + '%')
                    f.write('\nConfusion matrix:\n' + str(m_confusion_multi))

                print("\n###\n###\nTest accuracy: b_acc = %.2f%% | m_acc = %.2f%%" % (b_acc_bin, m_acc_bin))

            print("\n\n--------- E N D   O F   S E S S I O N ---------\n\n\n")
            with open(path0 + 'saveAccuracy.txt', 'a') as f:
                f.write("--------- E N D   O F   S E S S I O N ---------\n\n\n")

            with open(path0 + 'number_of_execution.txt', 'w') as f:
                f.write(str(date) + '\n' + str(number_run))
