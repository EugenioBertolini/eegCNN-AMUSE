# eegCNN-AMUSE
- Go to http://bnci-horizon-2020.eu/database/data-sets and download 1 or more files from "18. AMUSE - an auditory speller based on spatial hearing (009-2015)"
- myDataPrep.m: Use my Matlab function to genarate and save the filters' coefficients and visualize them. Make sure that the saved filename and the saved variable names have the same name of the ones that you can find in the python files). I also uploaded the .mat file containing the filters' coefficient for attenuation 80dB and ripple 80dB (the best I can get before before Matlab crush), and some chebyshev type II filters generated with different orders.
- Move the import functions in the right folder (or adjust the path names in python).
- myMain.py: this is the main code. Check it complitely before each run. The code is able to store the results automatically in many forms. Take a notebook and a pen and write the paths down, so you make sure you don't miss any. If you are using Google Colab, you can just call myMain.py to execute the entire code.
- myCNN.py: this contains the CNN from https://github.com/vlawhern/arl-eegmodels written by https://github.com/vlawhern and a function to retrive the best epoch of your training, copied from https://github.com/tensorflow/tensorflow/issues/35634 and written by https://github.com/iliaschalkidis.
- myFunc.py: it contains the function to select the optimizer and also the funtions to generate (and cut, shift, normalize and filter) the ERP datasets for the CNN.
- myPlots.py: it contains the functions used to plot the loss and accuracy for each epoch of the training (train and validation curve), and the function to plot the confusion matrices nicely.
- myLDA.m and myLDAexample.m: apply the SWLDA to obtain a classification. Play around with the p_valueIN and the p_valueOUT and check which channels are the most important according to the stepwise method for feature selection.
