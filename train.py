#!/usr/bin/env python

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import tensorflow as tf
tf.config.optimizer.set_jit(True) # This increases training time slightly

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import random

# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# Global vars
dat_dir = "../Train_data" 
image_size = (128, 128) # 128x128 image size
n_classes = 3
batch_n = 32


def load_images():
    # Data Gen to add randomness to images
    dg = ImageDataGenerator(rescale = 1. / 255, shear_range = 0.1,
                            zoom_range = 0.2, rotation_range = 90,
                            width_shift_range = 0.33, height_shift_range = 0.33,
                            horizontal_flip = True, vertical_flip = True,
                            brightness_range = (0.75, 1.25),
                            validation_split = 0.2)
    
    # Get training set
    X_train = dg.flow_from_directory(dat_dir, target_size = image_size,
                                     classes = ['cherry', 'strawberry', 'tomato'], 
                                     batch_size = batch_n, shuffle = True,
                                     class_mode = 'categorical', subset='training')
    
    # Get validation set
    X_valid = dg.flow_from_directory(dat_dir, target_size = image_size,
                                     classes = ['cherry', 'strawberry', 'tomato'], 
                                     batch_size = batch_n, shuffle = False,
                                     class_mode = 'categorical', subset='validation')


    return X_train, X_valid 


def construct_model():
    """
    Construct the CNN model.
    :return: model
    """
    # Base Model
    model = Sequential()
    
    # CONV(16, RELU) > CONV(32, RELU) > POOL
    model.add(Conv2D(filters = 16, kernel_size = 3, input_shape=(128, 128, 3), activation= 'relu', padding='same'))
    model.add(Conv2D(filters = 32, kernel_size = 3, activation= 'relu', padding='same'))
    model.add(MaxPool2D(pool_size = 2, strides = 2))
    
    # CONV(64, RELU) > CONV(64, RELU) > CONV(64, RELU) > POOL
    model.add(Conv2D(filters = 64, kernel_size = 3, activation= 'relu', padding='same'))
    model.add(Conv2D(filters = 64, kernel_size = 3, activation= 'relu', padding='same'))
    model.add(Conv2D(filters = 64, kernel_size = 3, activation= 'relu', padding='same'))
    model.add(MaxPool2D(pool_size = 2, strides = 2))
    
    # CONV(128, RELU) > POOL
    model.add(Conv2D(filters = 128, kernel_size = 3, activation = 'relu', padding='same')) 
    model.add(MaxPool2D(pool_size = 2, strides = 2))
    
    # Dropout, Flatten, Dense, Classify
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(300, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation = 'softmax'))
    
    # Compile model
    opt = keras.optimizers.Adam(learning_rate = 0.0001) 
    model.compile(loss = 'categorical_crossentropy',
              optimizer = opt,
              metrics = ['accuracy'])
    
    # Print Model
    print(model.summary())
    
    return model


def construct_mlp():
    # Base model
    mlp = Sequential()    

    # Add layers
    mlp.add(Dense(16, activation='relu', input_shape = (128, 128, 3)))    
    mlp.add(Dense(32, activation = 'relu'))
    mlp.add(Dense(64, activation = 'relu'))
    
    # Flatten, Classify
    mlp.add(Flatten())
    mlp.add(Dense(3, activation = 'softmax')) 

    # Compile model
    opt = keras.optimizers.SGD(learning_rate = 0.001) # Set optimizer 
    mlp.compile(loss = 'categorical_crossentropy',
              optimizer = opt,
              metrics = ['accuracy'])
    
    print(mlp.summary())
    
    return mlp

def plot_metrics(history):
    # Accuracy over epoch
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc = 'upper left')
    plt.show()
    # Loss over epoch
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc = 'upper left')
    plt.show()
    return

def train_model(model, X_train, X_valid):
    """
    Train the CNN model
    ***
        Please add your training implementation here, including pre-processing and training
    ***
    :param model: the initial CNN model
    :return:model:   the trained CNN model
    """
    # Set early stopping    
    stop = EarlyStopping(monitor = 'val_loss', patience = 4)
    
    # Model Checkpoint, this will only save the best model
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath = "model/model_new.h5",
    monitor = 'val_loss',
    save_best_only = True)
    
    # Set adaptive learning rate
    lr_reduce = ReduceLROnPlateau(monitor = 'val_loss', 
                                                patience = 2, 
                                                verbose = 1, 
                                                factor = 0.25, 
                                                min_lr = 0.000001)
       
    # Fit Model
    model = model.fit(X_train, validation_data = X_valid,
                        steps_per_epoch = X_train.samples // batch_n,
                        validation_steps = X_valid.samples // batch_n,
                        callbacks = [stop, checkpoint, lr_reduce], epochs = 200) 
       
    return model


def train_mlp(model, X_train, X_valid):
    """
    Train the MLP model
    ***
        Please add your training implementation here, including pre-processing and training
    ***
    :param model: the initial CNN model
    :return:model:   the trained CNN model
    """
    # This saves the model
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath = "model/model_mlp.h5")
     
    # Fit Model
    model = model.fit(X_train, validation_data = X_valid, epochs = 25,
                      callbacks = [checkpoint]) 
       
    return model

if __name__ == '__main__':
  
    # Load images
    X_train, X_valid = load_images()
        
    # Build model
    #mlp = construct_mlp()
    model = construct_model()
    
    # Train model
    #mlp = train_mlp(mlp, X_train, X_valid)
    model = train_model(model, X_train, X_valid)
    
    # Plot model accuracy/loss
    #plot_metrics(mlp)   
    plot_metrics(model)