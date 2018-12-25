import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn import mixture 
from feature_extraction import extract_features
#from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")
#path to training data
# source   = "development_set/"
source   = "trainingData/"   
#path where training speakers will be saved


dest = "Speakers_models/"
train_file = "trainingDataPath.txt"        
file_paths = open(train_file,'r')
count = 1
# Extracting features for each speaker (5 files per speakers)
features = np.asarray(())
for path in file_paths:    
    path = path.strip()   
    print ("path is : ",path)
  
    # read the audio
    sr,audio = read(source + path)
    
    # extract 40 dimensional MFCC & delta MFCC features
    vector   = extract_features(audio,sr)
    print("vector size is : ", vector.shape)
    print("vector is :",vector)
    
    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector)) # add new feature add in old feature vertically using VSTACK
    # when features of 5 files of speaker are concatenated, then do model training
	# -> if count == 5: --> edited below
    print("size of feature is : ", features.shape)
    print("feature is :", features)
    '''if count == 2: # no of input user    
        gmm = mixture.GaussianMixture(n_components = 16, max_iter = 200, covariance_type='diag',n_init = 3)
        gmm.fit(features)
        
        # dumping the trained gaussian model
        picklefile = path.split("-")[0]+".gmm"
        cPickle.dump(gmm,open(dest + picklefile,'wb'))
        print ('+ modeling completed for speaker:',picklefile," with data point = ",features.shape)    
        features = np.asarray(())
        print("if count 2 that Feature is: ",features)
        count = 0
    count = count + 1'''
