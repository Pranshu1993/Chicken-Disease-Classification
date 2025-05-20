print("DEBUG: prepare_base_model.py loaded")

import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,                              # already (224, 224, 3)       
            include_top=self.config.params_include_top,
            weights=self.config.params_weights
        )

        PrepareBaseModel.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:                                                                  # Freeze all layers by making them untrainable
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
                for layer in model.layers[:-freeze_till]:
                    model.trainable = False
        flatten_in = tf.keras.layers.Flatten()(model.output)
        predictions = tf.keras.layers.Dense(
                classes, 
                activation='softmax'
                )(flatten_in)
        full_model = tf.keras.models.Model(
                inputs=model.input, 
                outputs=predictions)

        full_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                loss='categorical_crossentropy', 
                metrics=['accuracy']
                )
        
        full_model.summary()
        return full_model

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=self.config.params_freeze_all,
            freeze_till=self.config.params_freeze_till,
            learning_rate=self.config.params_learning_rate
        )
        
        PrepareBaseModel.save_model(path=self.config.updated_base_model_path, model=self.full_model)      #static method is called using classname.method(). No instance is needed.
    

    def save_model(path: str, model: tf.keras.Model):
        model.save(path)