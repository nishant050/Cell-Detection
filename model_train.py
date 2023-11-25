from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from skimage.io import imread
from sklearn.preprocessing import StandardScaler
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
import time
import mahotas as mh
from mahotas.features import surf
from sklearn.decomposition import PCA







class Model_Trainer:
    def __init__(self, positive_images_path, negative_images_path, test_mode = False, custom_images = False):
        #Loading images
        if not custom_images:
            self.pos_im_path = positive_images_path
            self.neg_im_path = negative_images_path

            self.pos_im_listing = os.listdir(self.pos_im_path)
            self.neg_im_listing = os.listdir(self.neg_im_path)
            self.num_pos_samples = size(self.pos_im_listing)
            self.num_neg_samples = size(self.neg_im_listing)
            print('Here are the number of images found: \n')
            print('No of positive samples: ', self.num_pos_samples)
            print('No of negative sample: ', self.num_neg_samples)
            if not test_mode:
                user_response = input('Do you want to change no. of pos and neg samples: \nReply with y or n')

                if user_response == 'y':
                    pos_sample = int(input('Enter no. of positive samples: '))
                    neg_sample = int(input('Enter no. of negative samples: '))

                    self.pos_im_listing = self.pos_im_listing[:pos_sample]
                    self.neg_im_listing = self.neg_im_listing[:neg_sample]
                else:
                    print('All pos and neg image path loaded\n')
            else:
                self.pos_im_listing = self.pos_im_listing[:10]
                self.neg_im_listing = self.neg_im_listing[:10]
                print('Testing mode is ON: ')
                print('No of positive samples: ', size(self.pos_im_listing))
                print('No of negative sample: ', size(self.neg_im_listing))
        
        else:
            self.pos_im_listing = positive_images_path
            self.neg_im_listing = negative_images_path
        
        #These variables are used for storing images and labels
        self.data = []
        self.labels = []
        self.processed_data = []
        self.hog_images = []
        self.scaler = None
        self.model = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        
        
    def create_image_dataset(self, custom_images = False, resize = True):
        print('Creating image dataset')
        if not custom_images:
            for file in self.pos_im_listing:
                img = imread(self.pos_im_path + '\\' + file)
                if resize:
                    img = cv2.resize(img,(64,64))
                img = color.rgb2gray(img)
                self.data.append(img)
                self.labels.append(1)

            for file in self.neg_im_listing:
                img= imread(self.neg_im_path + '\\' + file)
                if resize:
                    img = cv2.resize(img,(64,64))
                img = color.rgb2gray(img)
                self.data.append(img)
                self.labels.append(0)
        else:
            for file in self.pos_im_listing:
                img = imread(file)
                if resize:
                    img = cv2.resize(img,(64,64))
                img = color.rgb2gray(img)
                self.data.append(img)
                self.labels.append(1)

            for file in self.neg_im_listing:
                img= imread(file)
                if resize:
                    img = cv2.resize(img,(64,64))
                img = color.rgb2gray(img)
                self.data.append(img)
                self.labels.append(0)
            
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        print('Image Dataset created, stored in variable "self.data"')
        
    def extract_features(self,params = None, method = 'hog'):
        """
        For HOG: 
        params = {
          'number_of_orientations': 9,        # 6 - 12
          'pixels_per_cell': (8,8),                # 8, 16
          'cells_per_block': (2,2),                # 1, 2
          'do_transform_sqrt': True
                    }
        """
        print('Extracting features started')
        if method == 'hog':
            for image in self.data:
                features, hog_images = hog(image,
                                        orientations = params['number_of_orientations'],
                                        pixels_per_cell = params['pixels_per_cell'],
                                        cells_per_block = params['cells_per_block'],
                                        transform_sqrt = params['do_transform_sqrt'],
                                        visualize = True)
        
                self.hog_images.append(hog_images)
            
                self.processed_data.append(features)
            self.hog_images = np.array(self.hog_images)
            self.processed_data = np.array(self.processed_data)
                        
        elif method == 'surf':
            n = 1
            pca = PCA(n_components=n)
            new_labels = []
            count = 0
            for image in self.data:
                image = cv2.resize(image*255, (200,200))
                spoints = surf.surf(image, 4, 6, 2)
                if spoints.shape[0] >= n:
                    reduced_data = pca.fit_transform(spoints.T)
                    reduced_data = reduced_data.T
                    self.processed_data.append(reduced_data)
                    if self.labels[count] == 1:
                        new_labels.append([[1]]*n)
                    else:
                         new_labels.append([[0]]*n)
                count += 1
            self.labels = np.array(new_labels)        
            self.processed_data = np.array(self.processed_data)
            self.processed_data = np.reshape(self.processed_data, (-1,70))
            self.labels = np.reshape(self.labels, (-1,1))
            self.labels = np.squeeze(self.labels)
            print(self.labels.shape)
            print(self.processed_data.shape)
        self.scaler = StandardScaler().fit(self.processed_data)
        self.processed_data = self.scaler.transform(self.processed_data)

        (self.x_train, self.x_test, self.y_train, self.y_test) = train_test_split(
            self.processed_data, self.labels, test_size=0.20, random_state=42)
        print('Features extracted and saved in following urls: \nself.hog_images\nself.processed_data\nself.x_train etc.')
            
    def model_train(self, load_model = False, model_path = None, auto = False):
    
        
        if load_model:
            self.model = job_lib.load(model_path)
        else:
            self.model = LinearSVC()
            
        if auto == True:
            self.create_image_dataset()
            params = {
                'number_of_orientations': 9,        # 6 - 12
                'pixels_per_cell': (8,8),                # 8, 16
                'cells_per_block': (2,2),                # 1, 2
                'do_transform_sqrt': True
            }
            
            self.extract_features(params)
        
        print(" Training Linear SVM classifier...")
        t_start = time.time()
        self.model.fit(self.x_train, self.y_train)
        time_taken = np.round((time.time() -  t_start), 2 )
        print(' Training Completed, Time taken is : ', time_taken)
        
        print(" Evaluating classifier on test data ...")
        predictions = self.model.predict(self.x_test)
        print(classification_report(self.y_test, predictions))
        
    def save_models(self, classifier_name, scaler_name):
        
        joblib.dump(self.model, f'{classifier_name}.npy')
        joblib.dump(self.scaler, f'{scaler_name}.npy')
        
    def load_models(self, classifier_name, scaler_name):
        self.model = joblib.load(f'{classifier_name}.npy')
        self.scaler = joblib.load(f'{scaler_name}.npy')
        
    

            
            
    
        
    
            
    

        
        
        