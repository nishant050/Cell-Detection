from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from skimage.io import imread
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from skimage import color
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import argparse
import cv2
import os
import glob
from PIL import Image # This will be used to read/modify images (can be done via OpenCV too)
from numpy import *
import matplotlib.pyplot as plt


class hog_helper:
    def __init__(self, image_path):
        
        print('.........hog_helper created.........\n\n')
        self.img = None
        self.img_size = None
        #self.resized = None
        self.image_path = image_path
        self.gray_img = None
        self.fd = None
        self.model = None
        self.result = None
        self.confidence = None
    def load_image(self):
        self.img = imread(self.image_path)
        self.img_size = self.img.shape
        
    #def resize(self, img_size = (64,64)):
        #self.resized = cv2.resize(np.array(self.img), img_size)
    
    def hog(self, orientations = 9, pixels_per_cell = (8, 8), cells_per_block = (2, 2), feature_vector=True, resize = True):
        #if resize:
            #self.resize()
        if len(self.img_size) > 2: 
            self.gray_img = color.rgb2gray(self.img)
            
        else:
            self.gray_img = np.array(self.img)
        
        
        self.fd = hog(self.gray_img, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector = feature_vector)
        self.fd = np.expand_dims(self.fd, axis = 0)
        
    def load_model(self, model_location = 'model3.npy' ):
        self.model = joblib.load(model_location)
    
    def predict(self, print_results = True):
        if self.model == None:
            print("Load the Model first using self.load_model( path of model )")
            print('Or you can load the model here \n')
            user_response = input('Press "y" to load the model:')
            if user_response == 'y':
                model_location = input('Enter the Input Location: \n')
                self.load_model(model_location)
            else:
                return None
        self.load_image()
        if len(self.img.shape) != 3 or self.img.shape[2] != 3:
            raise Exception("Image does not have 3 channels")
        self.hog()
        
        result = self.model.predict(self.fd)
        self.result = result[0]
        self.confidence = self.model.decision_function(self.fd)
        if print_results:
            if self.result == 1:
                print('The image has cell\n')
                print('Confidence score is:\n', self.confidence)
                self.show(fig_size=(2,2))

            else:
                print('No Cell in image')
                print('Confidence score is:\n', self.confidence)
                self.show(fig_size=(2,2))
        #return self.result
        else:
            return self.result, self.confidence, self.img
    def show(self, axis = 'off', fig_size = (5,5)):
        self.load_image()
        plt.figure(figsize = fig_size)
        plt.imshow(self.img)
        if axis == 'off':
            plt.axis('off')
        plt.show()
        
        
            
        
        
