# src/model_factory.py
import pandas as pd
import numpy as np
import xgboost as xgb
import os
from pathlib import Path

# Constraints
CPU_COUNT = os.cpu_count() or 1
os.environ["OMP_NUM_THREADS"] = str(CPU_COUNT)
os.environ["MKL_NUM_THREADS"] = str(CPU_COUNT)

class MLStrategy:
    """
    Dynamically generated/optimized ML model factory.
    The LLM will rewrite this entire class when it decides to evolve the architecture.
    """
    def __init__(self):
        self.model = None
        self.features = ['open', 'high', 'low', 'close', 'volume']
    
    def preprocess(self, df: pd.DataFrame):
        # LLM can rewrite this to add technical indicators, Fourier transforms, etc.
        df = df.copy()
        return df[self.features]

    def train(self, df: pd.DataFrame):
        X = self.preprocess(df)
        y = np.where(df['close'].shift(-1) > df['close'], 1, -1)
        
        # Training using CPU only (XGBoost defaults to CPU if tree_method is 'hist'/'auto' 
        # and no CUDA found, but we enforce nthread)
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            nthread=CPU_COUNT,
            tree_method='hist' 
        )
        self.model.fit(X, y)
    
    def predict(self, df: pd.DataFrame):
        X = self.preprocess(df)
        return self.model.predict(X)
