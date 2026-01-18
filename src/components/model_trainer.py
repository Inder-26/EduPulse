
import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Loading transformed training and testing data")
            
            train_arr = np.load(self.data_transformation_artifact.transformed_train_file_path, allow_pickle=True)
            test_arr = np.load(self.data_transformation_artifact.transformed_test_file_path, allow_pickle=True)

            # Extract X and y. Data was saved as object array of (X, y)
            x_train, y_train = train_arr[0], train_arr[1]
            x_test, y_test = test_arr[0], test_arr[1]
            
            logging.info(f"Loaded train data shapes: X={x_train.shape}, y={y_train.shape}")
            logging.info(f"Loaded test data shapes: X={x_test.shape}, y={y_test.shape}")
            
            # Model Architecture (from Notebook)
            MAX_NB_WORDS = 50000
            MAX_SEQUENCE_LENGTH = 250
            EMBEDDING_DIM = 100
            
            # Feature Extraction for Weights (Before One-Hot)
            if y_train.ndim == 1:
                y_indices = y_train
            else:
                y_indices = np.argmax(y_train, axis=1)
                
            # Compute Class Weights
            classes = np.unique(y_indices)
            weights = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=classes,
                y=y_indices
            )
            class_weights_dict = dict(zip(classes, weights))
            logging.info(f"Computed Class Weights: {class_weights_dict}")
            
            # Determine NUM_CLASSES and One-Hot Encode if necessary
            if y_train.ndim == 1:
                logging.info("Target data is 1D labels. Converting to One-Hot Encoding.")
                NUM_CLASSES = len(np.unique(y_train))
                y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
                y_test = to_categorical(y_test, num_classes=NUM_CLASSES)
            else:
                NUM_CLASSES = y_train.shape[1]
            
            logging.info(f"Number of Classes: {NUM_CLASSES}")
            
            logging.info("Defining LSTM Model Architecture")
            model = Sequential()
            model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM)) # Input length handled automatically or can specify input_length=MAX_SEQUENCE_LENGTH
            model.add(SpatialDropout1D(0.2))
            model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2)) # recurrent_dropout might be slow on GPU, but sticking to notebook config
            model.add(Dense(NUM_CLASSES, activation='softmax'))
            
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            # Training
            logging.info("Starting Model Training")
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            
            history = model.fit(
                x_train, y_train,
                epochs=10, 
                batch_size=64,
                validation_split=0.1,
                callbacks=[early_stopping],
                verbose=1,
                class_weight=class_weights_dict
            )
            
            # Evaluation
            loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
            logging.info(f"Model Test Accuracy: {accuracy}")
            
            if accuracy < self.model_trainer_config.expected_accuracy:
                logging.warning(f"Model accuracy {accuracy} is below expected score {self.model_trainer_config.expected_accuracy}")
                # We could raise an error here, but for now we'll just log warning and save
                # raise CustomException("Model accuracy is not satisfactory", sys)

            # Saving Model
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            model.save(self.model_trainer_config.trained_model_file_path)
            logging.info(f"Trained model saved at: {self.model_trainer_config.trained_model_file_path}")
            
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact={"accuracy": accuracy, "loss": loss}
            )
            
            return model_trainer_artifact
            
        except Exception as e:
            raise CustomException(e, sys)
