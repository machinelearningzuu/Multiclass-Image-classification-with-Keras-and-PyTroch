import tensorflow as tf
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator

from variables import*

def image_data_generator():
    train_datagen = ImageDataGenerator(
                                    rescale = rescale,
                                    shear_range = shear_range,
                                    zoom_range = zoom_range,
                                    horizontal_flip = True
                                    )
    test_datagen = ImageDataGenerator(rescale = rescale)


    train_generator = train_datagen.flow_from_directory(
                                    train_dir,
                                    target_size = target_size,
                                    color_mode = color_mode,
                                    batch_size = batch_size,
                                    classes = classes,
                                    shuffle = True)

    validation_generator = test_datagen.flow_from_directory(
                                    valid_dir,
                                    target_size = target_size,
                                    color_mode = color_mode,
                                    batch_size = valid_size,
                                    classes = classes,
                                    shuffle = True)

    test_generator = test_datagen.flow_from_directory(
                                    test_dir,
                                    target_size = target_size,
                                    color_mode = color_mode,
                                    batch_size = batch_size,
                                    classes = classes,
                                    shuffle = True)

    return train_generator, validation_generator, test_generator

