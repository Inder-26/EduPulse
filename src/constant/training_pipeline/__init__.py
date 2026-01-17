import os

TARGET_COLUMN = "rating"
PIPELINE_NAME: str = "EduPulse"
ARTIFACT_DIR: str = "artifacts"
FILE_NAME: str = "all_reviews_merged.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

# MongoDB Constants
DATA_INGESTION_COLLECTION_NAME: str = "allreviews"
DATA_INGESTION_DATABASE_NAME: str = "KrishNaikUdemyReviews"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2

# Model Trainer Constants
MODEL_TRAINER_DIR_NAME: str = 'model_trainer'
MODEL_TRAINER_TRAINED_MODEL_DIR: str = 'trained_model'
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.keras"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
