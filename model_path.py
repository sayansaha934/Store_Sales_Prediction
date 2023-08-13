from enum import Enum

class PreprocessingModelPath(str, Enum):
    INDEXER_MODEL = 'preprocessing_models/indexer_model'
    ENCODER_MODEL = 'preprocessing_models/encoder_model'
    IMPUTER_MODEL = 'preprocessing_models/imputer_model'
    SCALER_MODEL = 'preprocessing_models/scaler_model'
    CLUSTER_MODEL = 'preprocessing_models/cluster_model'

class PredictionModelPath(str, Enum):
    PATH = 'Models'