# CNN structure.
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

# Best Early Stopping Class
from tensorflow.keras.callbacks import EarlyStopping

# CNN function.
def EEGNet(nb_classes, Chans = 45, Samples = 256, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, kernLength2 = 16, norm_rate = 0.25, 
           dropoutType = 'Dropout', activFunct = 'softmax'):
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    input1   = Input(shape = (Chans, Samples, 1))
    ###########################################################################
    conv1        = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input1)
    batchnorm1   = BatchNormalization()(conv1)
    conv2        = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(batchnorm1)
    batchnorm2   = BatchNormalization()(conv2)
    activ2       = Activation('elu')(batchnorm2)
    avgpool2     = AveragePooling2D((1, 4))(activ2)
    dropout2     = dropoutType(dropoutRate)(avgpool2)
    ###########################################################################
    conv3        = SeparableConv2D(F2, (1, kernLength2),
                                   use_bias = False, padding = 'same')(dropout2)
    batchnorm3   = BatchNormalization()(conv3)
    activ3       = Activation('elu')(batchnorm3)
    avgpool3     = AveragePooling2D((1, 8))(activ3)
    dropout3     = dropoutType(dropoutRate)(avgpool3)
    ###########################################################################
    flatten      = Flatten(name = 'flatten')(dropout3)
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation(activFunct, name = 'softmax')(dense)
    return Model(inputs=input1, outputs=softmax)

def EEGNet3D(nb_classes, ChansX = 7, ChansY = 6, Samples = 256, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, kernLength2 = 16, norm_rate = 0.25, 
           dropoutType = 'Dropout', activFunct = 'softmax'):
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    input1   = Input(shape = (ChansX, ChansY, Samples))
    ###########################################################################
    conv1        = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (ChansX, ChansY, Samples),
                                   use_bias = False)(input1)
    batchnorm1   = BatchNormalization()(conv1)
    conv2        = DepthwiseConv2D((ChansX, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(batchnorm1)
    batchnorm2   = BatchNormalization()(conv2)
    activ2       = Activation('elu')(batchnorm2)
    avgpool2     = AveragePooling2D((1, 4))(activ2)
    dropout2     = dropoutType(dropoutRate)(avgpool2)
    ###########################################################################
    conv3        = SeparableConv2D(F2, (1, kernLength2),
                                   use_bias = False, padding = 'same')(dropout2)
    batchnorm3   = BatchNormalization()(conv3)
    activ3       = Activation('elu')(batchnorm3)
    avgpool3     = AveragePooling2D((1, 8))(activ3)
    dropout3     = dropoutType(dropoutRate)(avgpool3)
    ###########################################################################
    flatten      = Flatten(name = 'flatten')(dropout3)
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation(activFunct, name = 'softmax')(dense)
    return Model(inputs=input1, outputs=softmax)

class ReturnBestEarlyStopping(EarlyStopping):
    def __init__(self, **kwargs):
        super(ReturnBestEarlyStopping, self).__init__(**kwargs)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            if self.verbose > 0:
                print(f'\nEpoch {self.stopped_epoch + 1}: early stopping')
        elif self.restore_best_weights:
            if self.verbose > 0:
                print('Restoring model weights from the end of the best epoch.')
            self.model.set_weights(self.best_weights)
