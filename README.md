# eegCNN-AMUSE
- Go to http://bnci-horizon-2020.eu/database/data-sets and download 1 or more files from "18. AMUSE - an auditory speller based on spatial hearing (009-2015)"
- EEG_MatlabCode: Use my Matlab function to process the AMUSE EEG signals, so they look exactly as they should look. The variables' name is very important for the execution of the program, so be sure to check the code and modify what you need to modify (the filename and the saved variables name has to match the ones in the python code)
- Move the generated datasets in the right folder (or adjust the path names in python).
- Move the 2 import functions in the right folder (or adjust the path names in python).
- myCode.py: this is the main code. Check it complitely before each run. The code is able to store the results automatically in many forms. Take a notebook and a pen and write them down, so you make sure you don't miss any.
- myCNN.py: this contains the CNN from https://github.com/vlawhern/arl-eegmodels written by https://github.com/vlawhern and a function to retrive the best epoch of your training, copied from https://github.com/tensorflow/tensorflow/issues/35634 and written by https://github.com/iliaschalkidis.
- myPlots.py: it contains the function used to plot the loss and accuracy for each epoch of the training (train and validation curve), and the function to plot the confusion matrices nicely.

