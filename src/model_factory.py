# src/model_factory.py
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import warnings
warnings.filterwarnings("ignore")

CPU_COUNT = os.cpu_count() or 1
os.environ["OMP_NUM_THREADS"] = str(CPU_COUNT)
os.environ["MKL_NUM_THREADS"] = str(CPU_COUNT)

class MLStrategy:
    """
    AMT-compliant ML Strategy Factory
    Features: Sessional/Daily/Weekly VWAP + TPO/VAH/VAL/POC via Market Profile
    """
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            nthread=CPU_COUNT,
            tree_method='hist',
            objective='binary:logistic'
        )
    
    def _add_time_features(self, df: pd.DataFrame):
        df = df.copy()
        idx = pd.to_datetime(df.index)
        
        # Session labels (4x6h blocks: 05-11, 11-17, 17-23, 23-05)
        def get_session_label(ts):
            h = ts.hour
            if 5 <= h < 11: return 'S1_05-11'
            elif 11 <= h < 17: return 'S2_11-17'
            elif 17 <= h < 23: return 'S3_17-23'
            else: return 'S4_23-05'
        
        df['session_label'] = idx.map(get_session_label)
        df['daily_key'] = (idx - pd.Timedelta(hours=5)).date
        df['session_key'] = df['daily_key'].astype(str) + '_' + df['session_label']
        df['week_id'] = (idx - pd.Timedelta(days=2)).isocalendar().week
        return df
    
    def _add_vwap(self, df: pd.DataFrame):
        df = df.copy()
        
        # Sessional VWAP (reset every 6h block)
        df['pv'] = df['close'] * df['volume']
        df['vwap_sessional'] = df.groupby('session_key')['pv'].cumsum() / df.groupby('session_key')['volume'].cumsum()
        
        # Daily VWAP (reset 05:00 WIB)
        df['vwap_daily'] = df.groupby('daily_key')['pv'].cumsum() / df.groupby('daily_key')['volume'].cumsum()
        
        # Weekly VWAP (reset Tuesday 05:00 WIB)
        df['vwap_weekly'] = df.groupby('week_id')['pv'].cumsum() / df.groupby('week_id')['volume'].cumsum()
        
        return df
    
    def _add_tpo_levels(self, df: pd.DataFrame):
        """Calculate POC, VAH, VAL per sessional block using Market Profile library"""
        df = df.copy()
        
        # Market Profile per session
        def calc_profile(grp):
            if len(grp) < 3:
                return pd.Series({'poc': grp['close'].iloc[-1], 'vah': grp['close'].iloc[-1], 'val': grp['close'].iloc[-1]})
            try:
                from market_profile import MarketProfile
                mp = MarketProfile(grp[['high','low','close','volume']], tick_size=0.1)
                poc = mp.poc_price
                vah, val = mp.value_area[1], mp.value_area[0]
            except Exception:
                # Fallback to simple range if MP fails
                poc = grp['close'].median()
                vah = grp['high'].quantile(0.85)
                val = grp['low'].quantile(0.15)
            return pd.Series({'poc': poc, 'vah': vah, 'val': val})
        
        profiles = df.groupby('session_key').apply(calc_profile)
        df = df.join(profiles, on='session_key')
        
        # Forward fill VAH/VAL within session so every bar has values
        df['vah'] = df.groupby('session_key')['vah'].ffill()
        df['val'] = df.groupby('session_key')['val'].ffill()
        df['poc'] = df.groupby('session_key')['poc'].ffill()
        
        return df
    
    def _feature_engineering(self, df: pd.DataFrame):
        """Combine all AMT features into ML-ready matrix"""
        df = self._add_time_features(df)
        df = self._add_vwap(df)
        df = self._add_tpo_levels(df)
        
        # Derived features
        df['dist_vwap_sess'] = df['close'] - df['vwap_sessional']
        df['dist_vwap_daily'] = df['close'] - df['vwap_daily']
        df['dist_vwap_weekly'] = df['close'] - df['vwap_weekly']
        df['dist_poc'] = df['close'] - df['poc']
        df['pos_in_va'] = (df['close'] - df['val']) / (df['vah'] - df['val'] + 1e-6)
        
        # Interaction features
        df['vwap_sess_vs_daily'] = df['vwap_sessional'] - df['vwap_daily']
        df['price_vs_poc_vwap'] = df['dist_poc'] * df['dist_vwap_sess']
        
        features = [
            'vwap_sessional', 'vwap_daily', 'vwap_weekly',
            'poc', 'vah', 'val',
            'dist_vwap_sess', 'dist_vwap_daily', 'dist_vwap_weekly',
            'dist_poc', 'pos_in_va',
            'vwap_sess_vs_daily', 'price_vs_poc_vwap'
        ]
        
        return df[features].fillna(0), features
    
    def train(self, df: pd.DataFrame):
        X, _ = self._feature_engineering(df)
        # Target: 1 if next close > current close, else -1
        y = np.where(df['close'].shift(-1) > df['close'], 1, 0)
        y = y[:-1]  # drop last NaN
        X = X.iloc[:-1]
        self.model.fit(X, y)
    
    def predict(self, df: pd.DataFrame):
        X, _ = self._feature_engineering(df)
        return self.model.predict(X)
