import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from utils.config import Config
from src.preprocessor import Preprocessor
from src.data_loader import load_raw_data, preprocess_datetime
from src.feature_builder import add_features

class Predictor:
    def __init__(self):
        self.model = load_model(Config.MODEL_SAVE_PATH)
        self.pre = Preprocessor()
        raw_df = load_raw_data()
        self.pre.scale_data(pd.DataFrame(), is_training=False)

    def predict_next_hours(self, last_window_data, n_steps=24, start_time=None):
        predictions = []
        num_features = len(Config.FEATURES)
        current_batch = last_window_data.reshape((1, Config.WINDOW_SIZE, num_features))
        res = self.model.predict(current_batch, verbose=0)
        predictions = res.reshape(-1, 1) 
        predictions_mw = self.pre.target_scaler.inverse_transform(predictions)

        return predictions_mw
