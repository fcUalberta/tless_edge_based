"""
Created on Wed Nov 20 10:27:07 2019

@author: frincy

Implements training a model on multiple classifiers and predicting on test set
The classifiers used are:
    1) Stochastic Gradient descent
    2) Perceptron
    3) Passive-Aggressive classifier with hinge loss
    4) Passive-Aggressive classifier with squared hinge loss
    
This is a sample train_test code speciically for feature_canny dataset
For training and testing on other datasets, the following needs to be altered
    1) All paths to input images and destination
    2) Preprocessing of test inputs in test() to make it similar to training set

"""

#import matplotlib.pyplot as plt

import pandas as pd
import cv2
import numpy as np


from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report
import dask.array as da

from dask_ml.model_selection import train_test_split
from dask_ml.wrappers import Incremental
from sklearn.utils import shuffle
import pickle
import statistics

import sys
import os


# Creating the list of Classifiers with their loss functions and other parameters
classifiers = [
    ("SGD Classifier", SGDClassifier(loss='log', penalty='l2', tol=0e-3)),
    ("Perceptron", Perceptron()),
    ("Passive-Aggressive I", PassiveAggressiveClassifier(loss='hinge',
                                                         C=1.0)),
    ("Passive-Aggressive II", PassiveAggressiveClassifier(loss='squared_hinge',
                                                          C=1.0)),
]


def create_labels(filepath):
    """
    Creates list of labels of training dataset from the annotations file
    
    Args:
        filepath: path of annotation files
    
    Returns: 
        label_df: Dataframe of labels
    """
    
    filelist = os.listdir(filepath)
    columns = ['filename','label']
    label_df = pd.DataFrame(columns = columns)
    count = 0
    col1 = []
    col2 = []
    
    for file in filelist:
        
        name = file[:-4]
        imagename = name+'.png'
        absolute_path = os.path.join(filepath,file)
    
        f = open(absolute_path,"r")
        classname = f.read(3).split(" ")
        print(classname)
        print(classname[0])
        
        col1.append(imagename)
        col2.append(classname[0])
        count += 1
        
       
    label_df = pd.DataFrame({'filename': col1, 'label': col2})    
    return label_df

    
def create_feature_matrix1(imagepath, H,W):
    """
    Creates feature matrix for all images in a dataset
    
    Args: 
        imagepath: path to the input images
        H,W: Height and width of output images
        
    Returns: 
        features_list: list of features of all input images
    """
    
    features_list = []
    imagelist =os.listdir(imagepath)

    print(len(imagelist))
    for image in imagelist:
        # load image
        img = cv2.imread(os.path.join(imagepath,image),0)
        img_resized = cv2.resize(img,(H,W)) # Resizing input image
        features = img_resized.flatten()
        features_list.append(features)

    return features_list

def train_batches(feature_matrix,y):
    """
    Trains a each dataset by 4 classifiers
    
    Args: 
        feature_matrix: list of features of input dataset
        y: labels of input dataset        
    
    """
    X_org,y_org = shuffle(feature_matrix, y, random_state=13)
    
    # Splitting 20% of dataset to be test later
    X_train_org, X_test_org, y_train_org, y_test_org = train_test_split(X_org, y_org,test_size=0.20)
    X = X_train_org
    y = y_train_org
    
    
    for name, clf in classifiers:
        inc = Incremental(clf, scoring='accuracy')
        batch_size=5000
        counter=0
        train_acc = []    
        test_acc = []
        
        # Initializing Standard Scaler and IPCA for each classifier
        
        SS = StandardScaler()
    #    IPCA = IncrementalPCA(n_components = 500)
        n=1
        print("Training ", name,".......\n")
        for j in range(80):
            if counter >= len(X):
                break
            
            # Splitting each batch into training and validation datset
            X_train, X_test, y_train, y_test = train_test_split(X[counter:counter+batch_size], y[counter:counter+batch_size],test_size=0.25)
            print("Iteration:",n)
        
            classes = da.unique(y_train).compute()
    
            # Feature Scaling
            SS.partial_fit(np.asarray(X_train))
            SS.transform(np.asarray(X_test))
            
            # Feature Decomposition
    #        IPCA.partial_fit(X_train)
    #        IPCA.transform(X_test)
            
            # Partial fitting - Stochastic Gradient Descent
            inc.partial_fit(X_train, y_train, classes=classes)
            print('Training Score:', inc.score(X_train, y_train))
            print('Validation Score:', inc.score(X_test, y_test))
            print("\n")
            
            # Concatenating batch scores
            train_acc.append(inc.score(X_train, y_train))     
            test_acc.append(inc.score(X_test, y_test))        
           
            if(len(X)-counter < batch_size):
                batch_size = len(X)-counter
            counter += batch_size
            n += 1
        
        
        # Savings the model
        filename = r'C:\PythonCodes\MM803\code\Outputs\New\f_canny_'+name+'.sav'
        pickle.dump(inc, open(filename, 'wb'))
        
        # Printing Model Accuracy
        print(name," MODEL ACCURACY")
        print("_______________________")
        print("Avg Training Accuracy of ", name,":", statistics.mean(train_acc))  
        print("Avg Test Accuracy ", name,":",statistics.mean(test_acc))
        
       
        # Testing on Unseen Data
        SS.transform(np.asarray(X_test_org[:5000]))
    #    IPCA.transform(X_test_org[:5000])
        print('\nFinal Testing Score on Unseen data 1 by ', name,':', inc.score(X_test_org[:10], y_test_org[:10]))
        print('Final Testing Score on Unseen data 2 by ', name,':', inc.score(X_test_org[10:100], y_test_org[10:100]))
        print('Final Testing Score on Unseen data 3 by ', name,':', inc.score(X_test_org[500:1000], y_test_org[500:1000]))
    
        print('\n\nClassification Report of', name)
        print('------------------------------------')
        print(classification_report(y_test_org[:5000],inc.predict(X_test_org[:5000]), digits = 4))
        print('====================================')
        print('\n')
    
    # Saving the trained StandardScaler to be used for testing
    filename_ss = r'C:\PythonCodes\MM803\code\Outputs\New\f_canny_SS.sav'
    pickle.dump(SS, open(filename_ss, 'wb'))
    
    # Saving the trained Incremental PCA to be used for testing
    #filename_ipca = r'C:\PythonCodes\MM803\code\Outputs\New\f_hed_IPCA.sav'
    #pickle.dump(IPCA, open(filename_ipca, 'wb'))  
    
def test():
    """
    Implements prediction of test set with different background
    on the trained model. This specific function is for CANNY dataset only.
    This function performs the initial preprocessing of test dataset to make it 
    similar to the dataset on which the model is trained on.
    
    To be used for another dataset, the test input needs to be preprocessed 
    according to that dataset.        
    """
    
    data = 'f_canny'
    dest = os.path.join(r"D:\Data\test1",data)
    
    if not os.path.exists(dest):
        os.makedirs(dest)
    
    # Loading the trained StandardScaler
    pickle_in2 = open(r'C:\PythonCodes\MM803\code\Outputs\New\f_canny_SS.sav',"rb")
    SS = pickle.load(pickle_in2)
    
    #pickle_in3 = open(r'C:\PythonCodes\MM803\code\Outputs\New\f_hed_IPCA.sav',"rb")
    #IPCA = pickle.load(pickle_in3)
    
    # insert at 1, 0 is the script path (or '' in REPL)
    sys.path.insert(1, r"C:\PythonCodes\MM803\code\feature_enhancement\hed")
    import feature_enhancement.feature_enhancement_feature_only  as features

    base = r"C:\PythonCodes\MM803\TestImages\cropped"
    testset = ['t1','t7','t10', 't16','t18','t27']
    for folder in testset:
        imagepath = os.path.join(base,folder)
        dest = os.path.join(base,"canny_"+folder)
        labels = pd.DataFrame(columns = ["filename","classname"])
        col1 = []
        col2 = []
        if not os.path.exists(dest):
            os.makedirs(dest)
        imagelist = os.listdir(imagepath)
        print((imagelist))
        for images in imagelist:
            imagename = os.path.join(imagepath,images)
               
            filename = images[:-4]
          
            # Reading and processing image
            image = cv2.imread(imagename,cv2.COLOR_BGR2GRAY)
    
            gray_img = cv2.imread(imagename,0)
            blurred = cv2.GaussianBlur(gray_img,(3,3),0)
            canny_edge = features.canny(image,blurred,alpha = 1, beta=0.6)
    
            cv2.imwrite(os.path.join(dest,filename+"_canny.png"),canny_edge)
    
            col1.append(imagename)
            col2.append(filename)
        feature_matrix_test = create_feature_matrix1(dest, 200,200)
        labels = pd.DataFrame({'filename': col1, 'label': col2}) 
  
        labels.to_csv(r"C:\PythonCodes\MM803\code\Mid_Outputs\test_" + folder+"_f_canny_labels.csv", sep='\t', encoding='utf-8', index = False, header = False)
        labels_df = pd.read_csv(r"C:\PythonCodes\MM803\code\Mid_Outputs\test_" + folder+"_f_canny_labels.csv", sep = '\t', header=None)
        y_test_new = labels_df[1]
    
        SS.transform(np.asarray(feature_matrix_test))
    ##    X_test_new = IPCA.transform(feature_matrix_test)
        X_test_new = feature_matrix_test
    
        for name, clf in classifiers:
            print("\nClassification by ", name, " on test set ", folder)
            print("_______________________________________")
            pickle_in1 = open(r'C:\PythonCodes\MM803\code\Outputs\New\f_canny_'+name+'.sav',"rb")
            inc = pickle.load(pickle_in1)
            
            for i in range(len(imagelist)):
                print('Score on Test Images:', imagelist[i][:-4]," ", inc.score(X_test_new[i:i+1], y_test_new[i:i+1]))
            
            print("Predicted Classes",inc.predict(X_test_new))
            print("Actual Classes \n\n", y_test_new)
        


if __name__ == '__main__':    
    """
    Main function of train_test module which implements training and 
    testing dataset with 4 classifier
    
    """

    
    base = r"C:\PythonCodes\MM803\without_noise\features_canny"
    
    imagepath = os.path.join(base,"images")
    filepath = os.path.join(base,"labels")
    
    # Step 1: Creating labels for training set (y values)
    print("Labels...:")
    labels = create_labels(filepath)
    
    # Saving labels as csv
    labels.to_csv(r"C:\PythonCodes\MM803\code\Mid_Outputs\f_canny_labels_wo_noise.csv", sep='\t', encoding='utf-8', index = False, header = False)
    labels_df = pd.read_csv(r"C:\PythonCodes\MM803\code\Mid_Outputs\f_canny_labels_wo_noise.csv", sep = "\t", header=None)
    
    print(labels_df.head())
    
    # Extracting just the classnames
    y = labels_df[1]
    print(y.head())
    
    # Step 2: Creating feature matrix from input images (X values)
    print("Feature Matrix...:")    
    feature_matrix = create_feature_matrix1(imagepath, 200,200)
    
    # Saving the feature_matrix in case the system hangs
    np.save(r"C:\PythonCodes\MM803\code\Mid_Outputs\FeatureMatrix_f_canny_wo",feature_matrix)
    feature_matrix = np.load(r"C:\PythonCodes\MM803\code\Mid_Outputs\FeatureMatrix_f_canny_wo.npy")
    
    # Step 3:  Training the input dataset in batches and testing the trained model on 
    train_batches(feature_matrix,y)
    test()
    
