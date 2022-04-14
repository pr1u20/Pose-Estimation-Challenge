#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 20:15:29 2021

@author: user
"""

from Keras_Generator import TF_Model


"""Train models with larger image size to get more accurate results.
This will also mean more computational time and effort."""

def main():
    
    """Define and train different models."""
    
    radiusss = [1]   # Radius of the section of image fed to the CNN
    img_size = (60, 60) # Size of the image
    
    for radius in radiusss:
    
        # Create or load model for specific image_size and radius
        Operation = TF_Model(img_size, radius)
        
        #Step 1: Take 100 random images 
        #Step 2: convert it into a dataset 
        #Step 3: train the model
        # Repeat the three steps n (epochs) number of times
        Operation.iterative_training(epochs = 2)
    

if __name__ == "__main__":

    main()
