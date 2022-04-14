#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 14:18:09 2021

@author: user
"""

from Keras_Generator import TF_Model
from utils import SatellitePoseEstimationDataset, Position, Score, Clean, orientation_error, threeshold_pixel, fill_spacecraft, borders_1, borders_2, borders_3
from submission import SubmissionWriter

import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import models
from scipy.optimize import minimize
import cv2


class Classify():
    
    """Make and visualize predictions."""
    
    def __init__(self, dataset_name = "validation"):
        size_unipix = (50, 50)
        size_imagenet = (224, 224)
        self.dataset_name = dataset_name
        
        self.dataset_root_dir ='speedplus/' # path to speed+'
        self.dataset_unipix = SatellitePoseEstimationDataset(root_dir=self.dataset_root_dir, newsize = size_unipix )
        self.dataset_imagenet = SatellitePoseEstimationDataset(root_dir=self.dataset_root_dir, newsize = size_imagenet)
        self.model_imagenet = models.load_model("obj/KERAS_ALL_imagenet")
        
        radius = 4
        self.model_inpix_4 = TF_Model((100,100), radius)
        
        radius = 1
        self.model_inpix_50_1 = TF_Model((50, 50), radius)
        
        radius = 1
        self.model_inpix_60_1 = TF_Model((60, 60), radius)
        
        self.score = Score()
        self.submission = SubmissionWriter()
        
    
    def predict_imagenet(self, im_num):
        
        """Use ResNet50 to make first estimates of position and oreintation."""
                
        im = np.array(self.dataset_imagenet.get_image(im_num, split = self.dataset_name, rgb =  True))
        im = preprocess_input(im)
        im = np.expand_dims(im, 0)
        pred_imagenet = self.model_imagenet.predict(im)[0]
        
        return pred_imagenet
        
        
    def predict_unitpix(self, im_num):
        
        """Use CNNs to segement the image with 1s and 0s."""
        
        unit_inpix_4 = self.model_inpix_4.classify_image(im_num, partition = self.dataset_name)
        unit_inpix_50_1 = self.model_inpix_50_1.classify_image(im_num, partition = self.dataset_name)
        unit_inpix_60_1 = self.model_inpix_60_1.classify_image(im_num, partition = self.dataset_name)
        
        avg_unipix = cv2.resize(unit_inpix_4, (50, 50)) + unit_inpix_50_1 + cv2.resize(unit_inpix_60_1, (50,50))

        unit_inpix_1 = borders_1(avg_unipix)
        unit_inpix_2 = borders_2(unit_inpix_50_1)
        unit_inpix_3 = borders_3(unit_inpix_60_1)
        
        unit_inpix = unit_inpix_1*2 + unit_inpix_2 + unit_inpix_3
        
        avg_unipix = unit_inpix*0.2 + cv2.resize(unit_inpix_4, (50, 50)) + unit_inpix_50_1 + cv2.resize(unit_inpix_60_1, (50,50))
        
        unit_inpix = threeshold_pixel(avg_unipix)
        
        unit_inpix = fill_spacecraft(unit_inpix)
        
        Clean_obj = Clean(unit_inpix)
        unit_inpix = Clean_obj.delete_small_obj(unit_inpix)
        unit_inpix = Clean_obj.correct_obj()
        
        return unit_inpix
    
    def find_optimum(self, im_num, epoch = 2):
        
        """Find the pose values that lead to the minimum of our defined loss function"""
        
        pred_imagenet = self.predict_imagenet(im_num)
        pred_unitpix = self.predict_unitpix(im_num)
        
        p = Position(pred_unitpix)

        guess_z = pred_imagenet[-1]
        guess_q = pred_imagenet[:4]
        
        bound = [(-1, 1), (-1, 1), (-1, 1), (-1, 1), (-0.66, 0.66), (-1.75, 1,75), (2, 10)]
        bound_q = [(-1, 1), (-1, 1), (-1, 1), (-1, 1)]
    
        
        def calculate_interpolation(arr_1, arr_2, z):
            #plt.imshow((arr_1 + arr_2))
            return np.sum(np.where((arr_1 + arr_2) != 1, 0, (arr_1 + arr_2))) + z**1.8
        
        def find_unit_q(x, show = False):
            q = x[:4]
            r = x[-3:]
            arr = self.dataset_unipix.guess_unit(q, r, show_im = show)
            return arr
        
        def loss_func_q(x):
            arr_1 = pred_unitpix
            arr_2 = find_unit_q(x)
            loss_value = calculate_interpolation(arr_1, arr_2, guess_z)
            return loss_value
        #COBYLA
        x_q = minimize(loss_func_q, np.concatenate((pred_imagenet[:4], p.XYZ_components(guess_z))), args=(), method='COBYLA' , jac=None, hess=None, hessp=None, bounds=bound_q, constraints=(), tol=None, callback=None, options=None)
        guess_q = x_q.x[:4] / np.sum(x_q.x[:4]**2)**(1/2)
        
        guess = np.concatenate((guess_q, x_q.x[-3:]))
        
        return guess[:4], guess[-3:]
    
    def visualize(self, q, r):
        
        """Visualize 2d image of 1s and 0s given pose parameters"""
        
        self.dataset_unipix.guess_unit(q, r, show_im = True)
        
    def append_submission(self, filename, q, r):
        
        """Append submission to the group of predicitions."""
        
        self.submission.append_real_test(filename, q, r, self.dataset_name)
        
    def evaluate(self, dataset = "validation", submit = False, show = False):
        
        """Process all images of specific images."""
        
        self.dataset_name = dataset
    
        print('Running evaluation on {} set...'.format(self.dataset_name))
        
        image_list = self.dataset_unipix.partitions[self.dataset_name]
    
        for num, image_name in enumerate(image_list):
            print(f"Image number: {num}")
            q, r = self.find_optimum(num, epoch = 2)
            print(f"q: {q}")
            print(f"r: {r}")
            
            
            if self.dataset_name == "validation":
                y_pred = np.concatenate((q, r))
                q_true, r_true = self.dataset_unipix.get_pose(num, partition = self.dataset_name)
                print(f"q_true: {q_true}")
                print(f"r_true: {r_true}")
                y_true = np.concatenate((q_true, r_true))
                self.score.append_score(y_pred, y_true)
                self.score.finalize()
            if submit == True: self.append_submission(image_name, q, r)
            if show == True:
                self.visualize(q, r)
                if self.dataset_name == "validation":
                    self.dataset_unipix.visualize(num, partition=self.dataset_name)

        

def main():
    
    """Evaluate datasets and export to csv file"""
    
    processing = Classify()
    processing.evaluate(dataset = 'sunlamp', submit = True, show = False)
    processing.evaluate(dataset = 'lightbox', submit = True, show = False)
    processing.submission.export(suffix='second_scipy_minimize')
    
    
if __name__ == "__main__":
    
    main()
        
    
            
        
            

