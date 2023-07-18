import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Conv2D, InputLayer, MaxPooling2D, GlobalAveragePooling2D
import os
import numpy as np

from utils.reading_data import load_example, read_data

class CNN:
    def __init__(self, input_shape, checkpoint_path, trained_model_dir, batch_size=32, epochs=20):
        self.input_size = input_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.checkpoint_dir = os.path.dirname(checkpoint_path)
        self.cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
        self.trained_model_dir = trained_model_dir
        self.model = self.build_model()
        self.compile()
        
    def build_model(self):
        model = Sequential()
        
        model.add(InputLayer(input_shape=self.input_size, batch_size=self.batch_size))
        
        model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu', name="conv_layer_1_1"))
        model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu', name="conv_layer_1_2"))
        model.add(MaxPooling2D((2, 2), 3, name="pool_layer_1"))

        model.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu', name="conv_layer_2_1"))
        model.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu', name="conv_layer_2_2"))
        model.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu', name="conv_layer_2_3"))
        model.add(MaxPooling2D((2, 2), 3, name="pool_layer_2"))

        model.add(Conv2D(256, (3, 3), strides=1, padding='same', activation='relu', name="conv_layer_3_1"))
        model.add(Conv2D(256, (3, 3), strides=1, padding='same', activation='relu', name="conv_layer_3_2"))
        model.add(Conv2D(256, (3, 3), strides=1, padding='same', activation='relu', name="conv_layer_3_3"))
        # model.add(MaxPooling2D((2, 2), 3, name="pool_layer_3"))

        # model.add(Conv2D(512, (3, 3), strides=1, padding='same', activation='relu', name="conv_layer_4_1"))
        # model.add(Conv2D(512, (3, 3), strides=1, padding='same', activation='relu', name="conv_layer_4_2"))
        # model.add(Conv2D(512, (3, 3), strides=1, padding='same', activation='relu', name="conv_layer_4_3"))
        # model.add(MaxPooling2D((2, 2), 3, name="pool_layer_3"))
        model.add(GlobalAveragePooling2D(name="GAP"))

        model.add(Dense(7, activation="softmax"))
        return model

    # def compile(self):
    #     self.model.compile(optimizer=tf.keras.optimizers.Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
    
    def summary(self):
        return self.model.summary()

    def read_data(self):
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = read_data()
        return True

    def compile(self):
        # if self.pretrained_model is not None:
        #     self.model.load_weights(self.pretrained_model)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                           loss="categorical_crossentropy",
                           metrics=['accuracy'])

    def train(self, val=None):
        if val is not None:
            self.model.fit(self.x_train,
                       self.y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       callbacks=self.cp_callback,
                       validation_data=(self.x_val, self.y_val))
        else:
            self.model.fit(self.x_train,
                       self.y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       callbacks=self.cp_callback)

    def evaluate(self):
        loss, metric = self.model.evaluate(self.x_test, self.y_test, verbose=1)
        if loss and metric:
            return loss, metric
        return False, False
        
    def predict(self):
        return np.argmax(self.model.predict(load_example()))
        
    def model_summary(self):
        return self.model.summary()

    def save_model(self):
        if self.model.save(self.trained_model_dir):
            return True
        return False

    def load_model(self):
        self.model = tf.keras.models.load_model(self.trained_model_dir)
        if self.model:
            return True
        return False
    
    def load_checkpoint(self):
        latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        check = self.model.load_weights(latest)
        return check