"""
Weather data processing utilities for rainfall prediction.
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from loguru import logger


class WeatherDataProcessor:
    """
    Comprehensive weather data processor for rainfall prediction tasks.
    """
    
    def __init__(self, lat: float = -33.8678, lon: float = 151.2073, 
                 timezone: str = "Australia/Sydney"):
        """
        Initialize WeatherDataProcessor.
        
        Args:
            lat: Latitude for weather data
            lon: Longitude for weather data  
            timezone: Timezone for weather data
        """
        self.lat = lat
        self.lon = lon
        self.timezone = timezone
        self.scaler = None
        self.imputer = None
        
    def fetch_weather_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch weather data from Open-Meteo API.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with weather data
        """
        daily_vars = [
            "weather_code",
            "temperature_2m_max", "temperature_2m_min",
            "apparent_temperature_max", "apparent_temperature_min",
            "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours",
            "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
            "shortwave_radiation_sum", "et0_fao_evapotranspiration",
            "sunshine_duration", "daylight_duration"
        ]
        
        url = (
            "https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={self.lat}&longitude={self.lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&daily={','.join(daily_vars)}"
            f"&timezone={self.timezone}"
        )
        
        logger.info(f"Fetching weather data from {start_date} to {end_date}")
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data["daily"])
        df["time"] = pd.to_datetime(df["time"]).dt.date
        
        logger.info(f"Fetched weather data with shape {df.shape}")
        return df
    
    def create_regression_target(self, df: pd.DataFrame, target_name: str = "precip_3day_next") -> pd.DataFrame:
        """
        Create regression target for 3-day precipitation prediction.
        
        Args:
            df: Input DataFrame with precipitation data
            target_name: Name of the target column
            
        Returns:
            DataFrame with target variable added
        """
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"])
        
        # Target: sum of next 3 days of precipitation
        df[target_name] = (
            df["precipitation_sum"].shift(-1) + 
            df["precipitation_sum"].shift(-2) + 
            df["precipitation_sum"].shift(-3)
        )
        
        # Drop rows where target can't be computed (end of dataset)
        df = df.dropna(subset=[target_name]).reset_index(drop=True)
        
        logger.info(f"Created regression target '{target_name}' with {len(df)} samples")
        return df
    
    def create_classification_target(self, df: pd.DataFrame, target_name: str = "target_rain", 
                                   threshold: float = 0.1, horizon_days: int = 7) -> pd.DataFrame:
        """
        Create classification target for rain prediction.
        
        Args:
            df: Input DataFrame with rain data
            target_name: Name of the target column
            threshold: Rain threshold in mm
            horizon_days: Prediction horizon in days
            
        Returns:
            DataFrame with target variable added
        """
        df = df.copy()
        df = df.sort_values("time").reset_index(drop=True)
        
        # Shift rain_sum by -horizon_days to get future rainfall
        df["rain_in_future"] = df["rain_sum"].shift(-horizon_days)
        
        # Apply threshold to create binary target
        df[target_name] = (df["rain_in_future"] > threshold).astype(int)
        
        # Drop rows with no future info (last horizon_days)
        df = df.dropna(subset=["rain_in_future"]).reset_index(drop=True)
        
        logger.info(f"Created classification target '{target_name}' with {len(df)} samples")
        return df
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features (year, month, season).
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with temporal features added
        """
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"])
        
        # Calendar parts
        df["year"] = df["time"].dt.year
        df["month"] = df["time"].dt.month
        
        # Southern Hemisphere seasons (AU)
        def au_season(m):
            if m in (12, 1, 2):  return "Summer"
            if m in (3, 4, 5):   return "Autumn"
            if m in (6, 7, 8):   return "Winter"
            return "Spring"      # 9,10,11
        
        df["season"] = df["month"].map(au_season)
        
        # Make season categorical with ordered categories
        df["season"] = pd.Categorical(df["season"],
                                    categories=["Summer","Autumn","Winter","Spring"],
                                    ordered=True)
        
        logger.info("Added temporal features: year, month, season")
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str, 
                          lags: List[int] = [1, 2]) -> pd.DataFrame:
        """
        Create lag features for temporal modeling.
        
        Args:
            df: Input DataFrame
            target_col: Target column to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features added
        """
        df = df.copy()
        
        for lag in lags:
            df[f"lag{lag}"] = df[target_col].shift(lag)
        
        logger.info(f"Created lag features for {target_col}: {lags}")
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, 
                               windows: List[int] = [3, 7, 14, 30]) -> pd.DataFrame:
        """
        Create rolling statistical features.
        
        Args:
            df: Input DataFrame
            windows: List of rolling window sizes
            
        Returns:
            DataFrame with rolling features added
        """
        df = df.copy()
        
        # Rolling precipitation features
        for window in windows:
            df[f"precip_{window}d_sum"] = df["precipitation_sum"].rolling(window, min_periods=1).sum()
            df[f"precip_{window}d_avg"] = df["precipitation_sum"].rolling(window, min_periods=1).mean()
            df[f"precip_{window}d_std"] = df["precipitation_sum"].rolling(window, min_periods=1).std()
            df[f"precip_{window}d_max"] = df["precipitation_sum"].rolling(window, min_periods=1).max()
            
            # Rain days count
            df[f"rain_days_{window}d"] = (
                df["precipitation_sum"].rolling(window, min_periods=1)
                .apply(lambda x: (x > 0).sum(), raw=True)
            )
        
        logger.info(f"Created rolling features for windows: {windows}")
        return df
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced meteorological features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with advanced features added
        """
        df = df.copy()
        
        # Temperature differentials and anomalies
        df['temp_range'] = df['temperature_2m_max'] - df['temperature_2m_min']
        df['apparent_temp_range'] = df['apparent_temperature_max'] - df['apparent_temperature_min']
        df['temp_anomaly'] = df['temperature_2m_max'] - df['temperature_2m_max'].rolling(30, min_periods=1).mean()
        
        # Advanced rolling statistics
        for col in ['temperature_2m_max', 'temperature_2m_min', 'wind_speed_10m_max', 'wind_gusts_10m_max']:
            for window in [3, 7, 14, 30]:
                df[f'{col}_{window}d_mean'] = df[col].rolling(window, min_periods=1).mean()
                df[f'{col}_{window}d_std'] = df[col].rolling(window, min_periods=1).std()
                df[f'{col}_{window}d_max'] = df[col].rolling(window, min_periods=1).max()
                df[f'{col}_{window}d_min'] = df[col].rolling(window, min_periods=1).min()
        
        # Weather pattern indicators
        df['humidity_proxy'] = (df['apparent_temperature_max'] - df['temperature_2m_max']).abs()
        df['storm_potential'] = (df['wind_speed_10m_max'] * df['wind_gusts_10m_max']) / 100
        df['atmospheric_instability'] = df['temp_range'] * df['wind_speed_10m_max']
        
        # Seasonal interactions and cyclical features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['temp_season_interaction'] = df['temperature_2m_max'] * df['month_sin']
        df['wind_season_interaction'] = df['wind_speed_10m_max'] * df['month_cos']
        
        # Atmospheric pressure and energy balance proxies
        df['pressure_proxy'] = df['shortwave_radiation_sum'] / (df['et0_fao_evapotranspiration'] + 1)
        df['energy_balance'] = df['shortwave_radiation_sum'] - df['et0_fao_evapotranspiration']
        
        # Wind direction features (cyclical encoding)
        df['wind_dir_sin'] = np.sin(2 * np.pi * df['wind_direction_10m_dominant'] / 360)
        df['wind_dir_cos'] = np.cos(2 * np.pi * df['wind_direction_10m_dominant'] / 360)
        
        # Weather code interactions
        df['weather_code_cloudy'] = (df['weather_code'] >= 2).astype(int)
        df['weather_code_rainy'] = (df['weather_code'] >= 61).astype(int)
        
        # Lag features (safe temporal features)
        for lag in [1, 2, 3, 7]:
            df[f'temp_max_lag{lag}'] = df['temperature_2m_max'].shift(lag)
            df[f'wind_speed_lag{lag}'] = df['wind_speed_10m_max'].shift(lag)
            df[f'radiation_lag{lag}'] = df['shortwave_radiation_sum'].shift(lag)
        
        logger.info("Created advanced meteorological features")
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                  categorical_cols: List[str] = ["weather_code", "season"]) -> pd.DataFrame:
        """
        Encode categorical features using one-hot encoding.
        
        Args:
            df: Input DataFrame
            categorical_cols: List of categorical columns to encode
            
        Returns:
            DataFrame with encoded categorical features
        """
        df = df.copy()
        
        for col in categorical_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                logger.info(f"Encoded categorical feature: {col}")
        
        return df
    
    def split_time_series_data(self, df: pd.DataFrame, target_col: str,
                              train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple:
        """
        Split time series data chronologically.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        df = df.copy()
        df = df.sort_values("time").reset_index(drop=True)
        
        # Remove target and time columns from features
        X = df.drop(columns=[target_col, "time"])
        y = df[target_col]
        
        # Chronological split indices
        n = len(df)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        train_idx = train_size
        val_idx = train_size + val_size
        
        # Split into sets
        X_train, y_train = X.iloc[:train_idx], y.iloc[:train_idx]
        X_val, y_val = X.iloc[train_idx:val_idx], y.iloc[train_idx:val_idx]
        X_test, y_test = X.iloc[val_idx:], y.iloc[val_idx:]
        
        logger.info(f"Split data - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                      X_test: pd.DataFrame, method: str = "standard") -> Tuple:
        """
        Scale features using specified method.
        
        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            method: Scaling method ("standard" or "minmax")
            
        Returns:
            Tuple of scaled (X_train, X_val, X_test)
        """
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
        
        # Copy dataframes
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        X_test_scaled = X_test.copy()
        
        # Replace inf values with NaN
        for df in [X_train_scaled, X_val_scaled, X_test_scaled]:
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fit scaler on training data
        self.scaler.fit(X_train_scaled)
        
        # Transform all sets
        X_train_scaled = pd.DataFrame(
            self.scaler.transform(X_train_scaled),
            index=X_train_scaled.index,
            columns=X_train_scaled.columns
        )
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val_scaled),
            index=X_val_scaled.index,
            columns=X_val_scaled.columns
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test_scaled),
            index=X_test_scaled.index,
            columns=X_test_scaled.columns
        )
        
        logger.info(f"Scaled features using {method} scaling")
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def impute_missing_values(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                             X_test: pd.DataFrame, strategy: str = "mean") -> Tuple:
        """
        Impute missing values using specified strategy.
        
        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            strategy: Imputation strategy ("mean", "median", "most_frequent")
            
        Returns:
            Tuple of imputed (X_train, X_val, X_test)
        """
        self.imputer = SimpleImputer(strategy=strategy)
        
        # Fit imputer on training data
        X_train_imputed = pd.DataFrame(
            self.imputer.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_val_imputed = pd.DataFrame(
            self.imputer.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        X_test_imputed = pd.DataFrame(
            self.imputer.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        logger.info(f"Imputed missing values using {strategy} strategy")
        return X_train_imputed, X_val_imputed, X_test_imputed
    
    def process_full_pipeline(self, start_date: str, end_date: str, 
                             task_type: str = "regression", 
                             target_name: str = None) -> Dict:
        """
        Run the complete data processing pipeline.
        
        Args:
            start_date: Start date for data fetching
            end_date: End date for data fetching
            task_type: Type of task ("regression" or "classification")
            target_name: Name of target column
            
        Returns:
            Dictionary with processed data splits
        """
        if target_name is None:
            target_name = "precip_3day_next" if task_type == "regression" else "target_rain"
        
        # Fetch data
        df = self.fetch_weather_data(start_date, end_date)
        
        # Create target
        if task_type == "regression":
            df = self.create_regression_target(df, target_name)
        else:
            df = self.create_classification_target(df, target_name)
        
        # Add temporal features
        df = self.add_temporal_features(df)
        
        # Create lag features
        if task_type == "regression":
            df = self.create_lag_features(df, target_name, [1, 2])
        else:
            df = self.create_lag_features(df, "rain_sum", [1, 2, 3, 7])
        
        # Create rolling features
        df = self.create_rolling_features(df)
        
        # Create advanced features
        df = self.create_advanced_features(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_time_series_data(df, target_name)
        
        # Impute missing values
        X_train, X_val, X_test = self.impute_missing_values(X_train, X_val, X_test)
        
        # Scale features
        X_train, X_val, X_test = self.scale_features(X_train, X_val, X_test)
        
        return {
            'X_train': X_train,
            'X_val': X_val, 
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': X_train.columns.tolist(),
            'target_name': target_name
        }
