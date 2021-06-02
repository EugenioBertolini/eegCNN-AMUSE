from tensorflow.keras.optimizers import Adam, RMSprop, SGD
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