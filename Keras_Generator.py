#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 11:50:30 2021

@author: user
"""

from utils import *
from tensorflow.keras import layers, Input, models


def shuffle_two_arrays(a, b):
    
    """Shuffle randomly two arrays in the same way"""
    
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    
    return np.array(a), np.array(b)

def init_model(im_shape):
    
    """Define the ML model."""
    
    inputs = Input(shape=im_shape, name="1")
    x = layers.Conv2D(32, (3, 3), input_shape= im_shape, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(16, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(30, activation='relu')(x)
    output = layers.Dense(1)(x)

    model = models.Model(inputs = inputs, outputs = output)
    model.summary()
    model.compile(optimizer='adam', loss='MSE', metrics=['binary_crossentropy', 'accuracy'])

    return model
    
class TF_Model():
    
    """Train model and predict images."""
    
    def __init__(self, imsize, radius):
        
        self.radius = radius
        self.shape = (2*radius+1, 2*radius+1, 1)
        self.imsize = imsize
        dataset_root_dir ='speedplus/' # path to speed+'
        self.dataset = SatellitePoseEstimationDataset(root_dir=dataset_root_dir, newsize = self.imsize)
        self.filepath = f"{imsize[0]}_{imsize[1]}_{radius}"
        
        try:
            self.model = models.load_model(f'obj/{self.filepath}')
            print(f"{self.filepath} downloaded succesfully!!")

        except OSError:
            self.model = init_model(self.shape)
            print(f"{self.filepath} does not exist --> New model created.")
    
    def train_model(self, X, Y):
        
        """Train the model."""
        
        self.model.fit(X, Y, batch_size = 64, epochs=1)
        
    def iterative_training(self, epochs = 1):
        
        """The dataset is very large so we will split it and train different sections. 
        This is not the best option, but it will allow us to keep training the models."""
        
        for epoch in range(epochs):
            
            print(f"Epoch: {epoch}")
            
            features, labels = self.dataset.generate_labeled_set(radius = self.radius, batch = 1000, partition = "train")
            
            features, labels = shuffle_two_arrays(features, labels)
            
            self.train_model(features, labels)
            
        self.model.save(f'obj/{self.filepath}')
        with open('obj/Models_Details.txt', "a") as f:
            f.write("\n")
            f.write(f"{self.filepath}\t{epochs}\t{self.imsize}\t{self.radius}")
            
            
    def classify_image(self, i, partition = "train"):
        
        """Process all image pixels."""
        
        im = np.array(self.dataset.get_image(i, split=partition))
        im_pad = np.pad(im, self.radius, mode='edge')
        
        n_features = (2*self.radius+1)  #(Total number of pixels)**(1/2) in the neighborhood
        feat_array = np.zeros((im.shape[0],im.shape[1],n_features, n_features))
        
        
        for x in range(self.radius,im_pad.shape[0]-(self.radius)):
            for y in range(self.radius,im_pad.shape[1]-(self.radius)):
                feat_array[x -self.radius,y-self.radius,:]=extract_neighborhood(x,y,im_pad,self.radius)
                
        all_pixels = feat_array.reshape(im.shape[0]*im.shape[1],n_features, n_features, 1)
        pred_pixels = self.model.predict(all_pixels)
        pred_image = pred_pixels.reshape(im.shape[0],im.shape[1])
        
        
        return pred_image