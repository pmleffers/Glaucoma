#!/usr/bin/env python3
from __future__ import print_function
import datetime
import os, sys, json, traceback, gzip
import pickle
import numpy as np
import random
import shutil
import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers import Input
from keras.models import Model
from keras.layers import add
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from sklearn.metrics import classification_report
import subprocess
from keras.regularizers import l2
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.utils import np_utils
from subprocess import call
import random
import imageio
import imgaug as ia
from keras.models import load_model
import warnings
warnings.filterwarnings("ignore")

# SageMaker paths
prefix      = '/opt/ml/'
input_path  = prefix + 'input/data/'
output_path = os.path.join(prefix, 'output')
model_path  = os.path.join(prefix, 'model')
param_path  = os.path.join(prefix, 'input/config/hyperparameters.json')
data_path = os.path.join(prefix, 'input/config/inputdataconfig.json')

def process_data(save_location='/opt/program/',script_name='preprocess.py'):
    #location of script and .zip file
    subprocess.call('python3 '+save_location+script_name,shell=True)
    with open(save_location+'save_records.pickle', 'rb') as records:
        save_records = pickle.load(records)

    subprocess.call('rm save_records.pickle',shell=True)
    train_images = save_records['train_len']
    validation_images = save_records['val_len']
    test_images = save_records['test_len']
    training_directory=save_records['train_dir']
    testing_directory=save_records['test_dir']
    validation_directory=save_records['val_dir']
    return train_images,validation_images,test_images,training_directory,testing_directory,validation_directory

def poly_decay(epoch,epochs_num=1,initial_learning_rate=0.002):
    max_epochs = epochs_num
    base_learning_rate = initial_learning_rate
    power = 1.0
    alpha = base_learning_rate*(1 - (epoch / float(max_epochs)))**power
    return alpha

def residual_module(input_data, kernel, stride, axis, reduce=False, reg_lambda=0.0001, epsilon=0.00002, momentum=0.9):
    shortcut = input_data
    bn1 = BatchNormalization(axis=axis, epsilon=epsilon, momentum=momentum)(input_data)
    act1 = Activation("relu")(bn1)
    conv1 = Conv2D(int(kernel * 0.25), (1, 1), use_bias=False,kernel_regularizer=l2(reg_lambda))(act1)

    bn2 = BatchNormalization(axis=axis, epsilon=epsilon,momentum=momentum)(conv1)
    act2 = Activation("relu")(bn2)
    conv2 = Conv2D(int(kernel * 0.25), (3, 3), strides=stride, padding="same", \
            use_bias=False, kernel_regularizer=l2(reg_lambda))(act2)

    bn3 = BatchNormalization(axis=axis, epsilon=epsilon, momentum=momentum)(conv2)
    act3 = Activation("relu")(bn3)
    conv3 = Conv2D(kernel, (1, 1), use_bias=False, kernel_regularizer=l2(reg_lambda))(act3)

    if reduce: 
        shortcut = Conv2D(kernel, (1, 1), strides=stride, use_bias=False, \
        kernel_regularizer=l2(reg_lambda))(act1)

    x = add([conv3, shortcut])
    return x

def build(width, height, depth, classes, stages, filters, reg_lambda=0.0001, epsilon=0.00002, momentum=0.9):
    input_shape = (height, width, depth)
    axis = -1

    if K.image_data_format() == "channels_first":
        input_shape = (depth, height, width)
        channel_dimension = 1

    inputs = Input(shape=input_shape)
    x = BatchNormalization(axis=axis, epsilon=epsilon,momentum=momentum)(inputs)
    x = Conv2D(filters[0], (5, 5), use_bias=False,padding="same", \
        kernel_regularizer=l2(reg_lambda))(x)
    x = BatchNormalization(axis=axis, epsilon=epsilon,momentum=momentum)(x)
    x = Activation("relu")(x)
    x = ZeroPadding2D((1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    for i in range(0, len(stages)):
        stride = (1, 1) if i == 0 else (2, 2)
        x = residual_module(x, filters[i + 1], stride,axis, reduce=True, \
            epsilon=epsilon, momentum=momentum)

        for j in range(0, stages[i] - 1):
            x = residual_module(x, filters[i + 1],(1, 1), axis, \
                epsilon=epsilon, momentum=momentum)

    x = BatchNormalization(axis=axis, epsilon=epsilon,momentum=momentum)(x)
    x = Activation("relu")(x)
    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)
    x = Dense(classes, kernel_regularizer=l2(reg_lambda))(x)
    x = Activation("softmax")(x)
    model = Model(inputs, x, name="resnet")
    return model

def train(batch_size=64, epochs=1, learning_rate=0.002,model_dir='/opt/ml/',input_data='missing'):
    #train,test,val: train, test, and validation directories
    img_width = 64
    img_height = 64
    model_name = 'glaucoma_keras_model.h5'
        
    if input_data=='missing':
        train_images,val_images,test_images,training_directory,testing_directory,validation_directory=process_data() 
        model_dir = '/opt/ml/model'
    else:  
        #example:
        #input_data={'train':'s3://my-bucket/my-training-data',
        #      'test':'s3://my-bucket/my-evaluation-data','val':'s3://my-bucket/my-evaluation-data'}
        training_directory,testing_directory,validation_directory=input_data['train'],input_data['test'],input_data['val']
        train_images,val_images,test_images=904,100,251

    training_augmentation = ImageDataGenerator(rescale=1/255.0, rotation_range=20, 
        zoom_range=0.05, width_shift_range=0.05,height_shift_range=0.05,
        shear_range=0.05,horizontal_flip=True,fill_mode="nearest")

    validation_augmentation = ImageDataGenerator(rescale=1/255.0)

    training_generator = training_augmentation.flow_from_directory(
        training_directory,class_mode="categorical",target_size=(img_height, img_width),
        color_mode="rgb",shuffle=True,batch_size=batch_size)

    validation_generator = validation_augmentation.flow_from_directory(
    validation_directory,class_mode="categorical",
        target_size=(img_height, img_width),color_mode="rgb",shuffle=False,batch_size=batch_size)

    testing_generator = validation_augmentation.flow_from_directory(testing_directory,
        class_mode="categorical",target_size=(img_height, img_width),color_mode="rgb",
        shuffle=False,batch_size=batch_size)

    model = build(img_height, img_width, 3, 2, (3, 4, 6), (64, 128, 256, 512), reg_lambda=0.0005)
    opt = SGD(lr=learning_rate, momentum=0.9)
    model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

    callbacks = [LearningRateScheduler(poly_decay)]
    train_model = model.fit_generator(training_generator, \
                  steps_per_epoch=train_images // batch_size, validation_data=validation_generator, \
                  validation_steps=val_images // batch_size, epochs=epochs, callbacks=callbacks)

    #if input_data=='missing':
    #    subprocess.call('rm -r '+'tmp',shell=True)

    model.save(model_dir+model_name)
    # save as JSON
    json_string = model.to_json()
    print('Saved trained model at %s ' % model_dir)


