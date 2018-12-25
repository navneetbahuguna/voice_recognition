import os
import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from feature_extraction import extract_features
#from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")
import time
"""
#path to training data
source   = "development_set/"   
modelpath = "speaker_models/"
test_file = "development_set_test.txt"        
file_paths = open(test_file,'r')
"""
#path to training data
source   = "SampleData/"   
#path where training speakers will be saved
modelpath = "Speakers_models/"
gmm_files = [os.path.join(modelpath,fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]
#Load the Gaussian gender Models
models = [cPickle.load(open(fname,'rb')) for fname in gmm_files]
speakers = [fname.split("/")[-1].split(".gmm")[0] for fname in gmm_files]
error = 0
total_sample = 0.0
print ("Do you want to Test a Single Audio: Press '1' or The complete Test Audio Sample: Press '0' ?")
take = int(input())
if take==1:
        try:
                print("Enter the File name from Test Audio Sample Collection :")
                path = input()
                print("Testing Audio",path)
                sr,audio=read(source + path)
                vector = extract_features(audio,sr)
                log_likelihood = np.zeros(len(models))
                for i in range(len(models)):
                        gmm= models[i]  #checking with each model one by one
                        print("GMM Is:",gmm)
                        scores = np.array(gmm.score(vector))
                        print("Scores is :",scores)
                        log_likelihood[i] = scores.sum()
                        print("Lof Likelihod is :",log_likelihood)
                winner = np.argmax(log_likelihood) # return max elementof the array in a particular axis
                print ("\tdetected as - ", speakers[winner])
                time.sleep(1.0)
        except Exception as e:
                print(e)
if take == 0:
        try:
                
                test_file = "testSamplePath.txt"        
                file_paths = open(test_file,'r')
                # Read the test directory and get the list of test audio files 
                for path in file_paths:
                        total_sample += 1.0
                        path = path.strip()
                        print ("Testing Audio : ", path)
                        sr,audio = read(source + path)
                        vector   = extract_features(audio,sr)
                        print("vector is :",vector)
                        log_likelihood = np.zeros(len(models))
                        print("log_likelihood is :",log_likelihood )
                        for i in range(len(models)):
                                gmm    = models[i]  #checking with each model one by one
                                scores = np.array(gmm.score(vector))
                                log_likelihood[i] = scores.sum()
                                print(" particular log_likelihood is :",log_likelihood )
                                
                        winner = np.argmax(log_likelihood)
                        print("\tdetected as - ", speakers[winner])
                        checker_name = path.split("_")[0]
                        if speakers[winner] != checker_name:
                                error += 1
                        time.sleep(1.0)
                print (error, total_sample)
                accuracy = ((total_sample - error) / total_sample) * 100
                print ("The Accuracy Percentage for the current testing Performance with MFCC + GMM is : ", accuracy, "%")
        except Exception as e:
                print(e)
print (" Mission Accomplished Successfully. ")

