import os
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
from src.exception import CustomException
from src.logger import logging
from src.constant.training_pipeline import TARGET_COLUMN

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting Data Transformation")
            
            # Load Data
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            
            logging.info("Loaded train and test data")
            
            # Fill missing values if any
            train_df = train_df.fillna("")
            test_df = test_df.fillna("")
            
            # Extract Text and Labels
            # Try 'content' first, fallback to 'review' if specific naming was used
            text_column = "content" if "content" in train_df.columns else "review"
            
            input_feature_train_arr = train_df[text_column].astype(str).tolist()
            input_feature_test_arr = test_df[text_column].astype(str).tolist()
            
            target_feature_train_arr = train_df[TARGET_COLUMN].tolist()
            target_feature_test_arr = test_df[TARGET_COLUMN].tolist()
            
            logging.info(f"Using text column: {text_column}")
            logging.info("Initializing Tokenizer")
            
            from tensorflow.keras.preprocessing.text import Tokenizer
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            
            # Tokenization
            vocab_size = 10000 
            embedding_dim = 16
            max_length = 100
            trunc_type = 'post'
            padding_type = 'post'
            oov_tok = "<OOV>"
            
            tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
            tokenizer.fit_on_texts(input_feature_train_arr)
            
            logging.info("Tokenizer fitted on training data")
            
            # Save Tokenizer
            os.makedirs(os.path.dirname(self.data_transformation_config.tokenizer_file_path), exist_ok=True)
            with open(self.data_transformation_config.tokenizer_file_path, 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
            logging.info(f"Tokenizer saved at {self.data_transformation_config.tokenizer_file_path}")
            
            # Text to Sequence & Padding
            training_sequences = tokenizer.texts_to_sequences(input_feature_train_arr)
            testing_sequences = tokenizer.texts_to_sequences(input_feature_test_arr)
            
            training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
            testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
            
            logging.info("Text sequences padded")
            
            # Encode Target
            label_encoder = LabelEncoder()
            training_labels = label_encoder.fit_transform(target_feature_train_arr)
            testing_labels = label_encoder.transform(target_feature_test_arr)
            # Ensure labels are numpy arrays
            training_labels = np.array(training_labels)
            testing_labels = np.array(testing_labels)

            # Combine Input and Target for saving (X, y)
            # Alternatively, save X and y separately or as a tuple object using numpy
            
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_file_path), exist_ok=True)
            
            # Saving as .npy files containing both X and y
            # We can save a dictionary or just save X and expect y to be handled differently.
            # For simplicity, let's save X and y together in a dictionary using np.savez or pickle. 
            # OR as per the config 'train.npy', sticking to a numpy array might be cleaner if we stack them?
            # Creating a dictionary for clarity in this specific NLP use case is safer than stacking int and float arrays.
            
            # However, standard practice often uses np.save with stacked array if dimensions match, 
            # OR using pickle for (X, y) tuple.
            # Given the config is .npy, let's use np.save on an object array or structured array?
            # Actually, `np.save` can save a dictionary if we allow pickle=True. 
            
            # Let's verify what ModelTrainer expects. We haven't written it yet.
            # Best way: Save X and y as a tuple object.
            
            train_arr = np.empty(2, dtype=object)
            train_arr[0] = training_padded
            train_arr[1] = training_labels

            test_arr = np.empty(2, dtype=object)
            test_arr[0] = testing_padded
            test_arr[1] = testing_labels
            
            np.save(self.data_transformation_config.transformed_train_file_path, train_arr)
            np.save(self.data_transformation_config.transformed_test_file_path, test_arr)
            
            logging.info("Transformed data saved")
            
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                tokenizer_file_path=self.data_transformation_config.tokenizer_file_path
            )
            
            return data_transformation_artifact
            
        except Exception as e:
            raise CustomException(e, sys)
