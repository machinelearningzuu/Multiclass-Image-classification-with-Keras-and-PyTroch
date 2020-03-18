import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import keras
import numpy as np
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Activation, Dense
from util import image_data_generator
from variables import *

class VGG16(object):
    def __init__(self):
        train_generator, validation_generator, test_generator = image_data_generator()
        self.test_generator = test_generator
        self.train_generator = train_generator
        self.validation_generator = validation_generator

    def model_conversion(self): #VGG16 is not build through sequential API, so we need to convert it to sequential
        vgg_functional = keras.applications.vgg16.VGG16()
        vgg_functional.summary()
        model = Sequential()
        for layer in vgg_functional.layers[:-1]:# remove the softmax in original model. because we have only 3 classes
            model.add(layer)
        model.summary()
        #make all the layers freeze and only train newly added softmax layer
        for layer in model.layers:
            layer.trainable = False
        model.add(Dense(num_classes, activation='softmax'))
        model.summary()
        self.model = model

    def train(self):
        self.model.compile(
                          optimizer='Adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy']
                          )
        self.model.fit_generator(
                          self.train_generator,
                          steps_per_epoch= train_step,
                          validation_data=self.validation_generator,
                          validation_steps= val_step,
                          epochs=epochs,
                          verbose=verbose
                        )

    def save_model(self):
        model_json = self.model.to_json()
        with open(model_architecture, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(model_weights)
        print("Model Saved")

    def load_model(self):
        vgg_functional = keras.applications.vgg16.VGG16()
        vgg_functional.summary()
        model = Sequential()
        for layer in vgg_functional.layers[:-1]:# remove the softmax in original model. because we have only 3 classes
            layer.trainable = False
            model.add(layer)
        model.add(Dense(num_classes, activation='softmax'))
        model.load_weights(model_weights)
        print("Model Loaded")

        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        self.model = model

    def predict(self):
        Predictions = self.model.predict_generator(self.test_generator,steps=test_step)
        P = np.argmax(Predictions,axis=1)
        loss , accuracy = self.model.evaluate_generator(self.test_generator, steps=test_step)
        print("test loss : ",loss)
        print("test accuracy : ",accuracy)
        print("Predictions : ",P)

if __name__ == "__main__":
    model = VGG16()
    model.model_conversion()
    if os.path.exists(os.path.join(os.getcwd(),model_weights)):
        model.load_model()
    else:
        model.train()
        model.save_model()
    model.predict()