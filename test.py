#!/usr/bin/env python

"""Description:
The test.py is to evaluate your model on the test images.
***Please make sure this file work properly in your final submission***

©2020 Created by Yiming Peng and Bing Xue
"""
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# You need to install "imutils" lib by the following command:
#               pip install imutils

import numpy as np
import random
import tensorflow as tf

# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
print(tf.version.VERSION)

def preprocess_data():
    
   # I use ImageDataGenerator to load in the images from the test folder
   # It rescales the images but performs no other transformations
   # Without this, test accuracy is very bad as the training and test images 
      # are of different types.
    
    # Create data generator
    dg = ImageDataGenerator(rescale = 1. / 255)    
    
    # Get path to test images
    dat_dir = "../ProjectTemplate_python3.8/data/test"
    
    # Get training set
    X = dg.flow_from_directory(dat_dir, target_size = (128, 128),
                                     classes = ['cherry', 'strawberry', 'tomato'], 
                                     batch_size = 16, shuffle = False,
                                     class_mode = 'categorical')
    
    return X


def evaluate(X_test):
    """
    Evaluation on test images
    ******Please do not change this function******
    :param X_test: test images
    :return: the accuracy
    """
    
    # Load Model
    model = load_model('model/model.h5') ## REMEMBER TO CHANGE BACK!!
    
    print(model.summary())
    
    return model.evaluate(X_test, verbose = 1)


if __name__ == '__main__':
    
    X_test = preprocess_data()
    
    loss, accuracy = evaluate(X_test)
    
    print("loss={}, accuracy={}".format(loss, accuracy))
