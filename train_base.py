
import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers 
from tensorflow.keras import initializers
from tensorflow.keras import callbacks
from tensorflow.keras.applications.vgg16 import VGG16 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt

import pandas as pd 
import numpy as np
import os
import cv2
from PIL import Image
import argparse

"""
    Trains a vgg16 model with the base layers and added linear layers to the top. Base model layers is frozen. Uses checkpoint callback to save the best model only.
    To run:
        python3 train_base.py --model_name "NAME" --epochs [int] --learning_rate [float] --batch_size [int] --path_dataset "PATH_TO_DATA" --path_model "PATH_SAVE_MODEL"
"""


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="001", help="Model name, used for saving it.")
parser.add_argument("--epochs", type=int, default=256, help="Amount of epochs to train the model for.")
parser.add_argument("--learning_rate", type=float, default=1e-05, help="Learning rate of the model/optimizer.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
parser.add_argument("--path_dataset", type=str, default=None, help="Path to dataset")
parser.add_argument("--path_model", type=str, default=None, help="Path to where to save model checkpoints and loss history")


args = parser.parse_args()
print("List of arguments:")
print(args)

#information about the model
model_name = args.model_name
epochs = args.epochs
learning_rate = args.learning_rate
batch_size = args.batch_size
model_location = args.path_model

#specify checkpoint location
checkpoint_filepath = os.path.join(model_location, model_name + "_checkpoint")

os.makedirs(checkpoint_filepath, exist_ok=True)

#metric to track (will be used for checkpoint)
metric = tensorflow.keras.metrics.AUC()
metric_name = metric.name

#dataset paths
splitted_data = args.path_dataset
path_train = splitted_data + '/train'
path_valid = splitted_data + '/valid'
#path_test = splitted_data + '/test' #unused

#create train and validation data generators & batches
#augmenting train data with ImageDataGenerator parameters
train_datagen = ImageDataGenerator(rescale= 1./255, 
                                   rotation_range= 40, 
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range= 0.2,
                                   horizontal_flip= True,
                                   fill_mode='nearest')

#validation data must not be augmented, just rescaled
validation_datagen = ImageDataGenerator(rescale=1./255)

train_batches = train_datagen.flow_from_directory(path_train,
                                                  color_mode='rgb',
                                                  target_size=(224, 224),
                                                  batch_size=batch_size,
                                                  class_mode='binary')

validation_batches = validation_datagen.flow_from_directory(path_valid, 
                                                            color_mode='rgb', 
                                                            target_size=(224,224),
                                                            batch_size=batch_size,
                                                            class_mode='binary')


#Load vgg16 model as base model and use imageNet weights
base_model = VGG16(weights='imagenet',
                   include_top= False,
                   input_shape=(224,224,3),
                   pooling="avg")

#freeze base model layers
base_model.trainable=False

#create a new sequential model and add the base model to it
model = models.Sequential()
for layer in base_model.layers:
    model.add(layer)
    if layer.name == "block4_pool":
        #we do not want to add the next convolutional block, so we stop adding the layers after this one
        break

#add our own linear (Dense) layers
model.add(layers.Flatten())
model.add(layers.Dense(units=512, activation="relu"))

model.add(layers.Dense(units=512, activation = "relu"))
model.add(layers.Dense(units=256, activation = "relu"))
model.add(layers.Dense(units=256, activation = "relu"))
#ouput layer with 1 unit for binary classification
model.add(layers.Dense(units=1, activation= 'sigmoid', name='predictions'))

#initialize checkpoint callback
model_checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor="val_"+metric_name,
    mode='max',
    save_best_only=True)

#compile and fit with Adam optimizer
model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=learning_rate), metrics=[metric])

history = model.fit(train_batches, epochs=epochs, validation_data=validation_batches, callbacks=[model_checkpoint_callback], verbose=1)

#save the performance history as csv
hist_df = pd.DataFrame(history.history)
hist_csv_file = os.path.join(model_location, model_name + ".csv")
with open(hist_csv_file, mode="w") as f: 
  hist_df.to_csv(f)