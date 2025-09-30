"""
Data loading and preprocessing utilities for NBA draft prediction.
"""

import os
import re
from pathlib import Path
from typing import Dict, Optional, Union, Tuple

import numpy as np
import pandas as pd
from loguru import logger


# Month to feet mapping for height parsing
MONTH_TO_FEET = {
    # Abbreviations
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
    # Full names
    'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
    'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
}


def parse_height_to_cm(value: Union[str, float, None]) -> float:
    """
    Convert height values to centimeters.
    
    Supports two formats:
    1. Feet/inches format: 6'11'' (regex: ^\\s*(\\d+)'\\s*(\\d{1,2})''\\s*$)
    2. Date format: "D-MMM" (e.g., 1-Jun -> 6'1'' ; 11-May -> 5'11'').
       Rule: day = inches (0..11) and month = feet (1..12).
    
    Args:
        value: Height value to convert
        
    Returns:
        Height in centimeters or np.nan if conversion fails
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan

    s = str(value).strip()
    if s == "":
        return np.nan

    # Case 1: Strict format 6'11''
    m = re.match(r"^\s*(\d+)'\s*(\d{1,2})''\s*$", s)
    if m:
        feet = int(m.group(1))
        inches = int(m.group(2))
        if 0 <= inches <= 11 and feet >= 0:
            return float(feet * 30.48 + inches * 2.54)
        return np.nan

    # Case 2: "D-MMM" or "D/MMM" or "D MMM"
    m2 = re.match(r"^\s*(\d{1,2})\s*[-/\s]\s*([A-Za-z]{3,9})\s*$", s)
    if m2:
        day = int(m2.group(1))
        mon = m2.group(2).lower()
        mon_key = mon[:3] if mon[:3] in MONTH_TO_FEET else mon
        if mon_key in MONTH_TO_FEET:
            feet = MONTH_TO_FEET[mon_key]
            inches = day
            if 0 <= inches <= 11:  # Only valid inches
                return float(feet * 30.48 + inches * 2.54)
        return np.nan

    return np.nan


class DataLoader:
    """
    Data loader for NBA draft prediction datasets.
    """
    
    def __init__(self, data_folder: Union[str, Path] = "../data/raw"):
        """
        Initialize DataLoader.
        
        Args:
            data_folder: Path to the raw data folder
        """
        self.data_folder = Path(data_folder)
        logger.info(f"DataLoader initialized with data folder: {self.data_folder}")
    
    def load_raw_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all CSV files in the data folder and process them.
        
        Ensures that the 'ht' column is read as text to avoid date interpretation.
        Converts 'ht' to centimeters in the same column if it matches supported formats.
        
        Returns:
            Dictionary mapping dataset names to DataFrames
        """
        datasets = {}
        
        if not self.data_folder.exists():
            raise FileNotFoundError(f"Data folder not found: {self.data_folder}")
        
        for filename in os.listdir(self.data_folder):
            if not filename.lower().endswith(".csv"):
                continue

            file_path = self.data_folder / filename

            # Read header to check if 'ht' column exists
            with open(file_path, encoding="latin-1") as f:
                first_line = f.readline()
            columns = [col.strip() for col in first_line.strip().split(",")]

            # Force reading as text for 'ht' to avoid date interpretation
            dtype = {"ht": str} if "ht" in columns else None

            df = pd.read_csv(
                file_path,
                encoding="latin-1",
                dtype=dtype,
                keep_default_na=True
            )

            # If 'ht' exists, convert to centimeters using the two rules
            if "ht" in df.columns:
                df = df.copy()
                df["ht"] = df["ht"].apply(parse_height_to_cm)

            key = os.path.splitext(filename)[0]
            datasets[key] = df
            logger.info(f"Loaded dataset '{key}' with shape {df.shape}")

        return datasets
    
    def load_processed_data(self, data_folder: Union[str, Path] = "../data/processed") -> Dict[str, pd.DataFrame]:
        """
        Load processed datasets (train, validation, test splits).
        
        Args:
            data_folder: Path to the processed data folder
            
        Returns:
            Dictionary with 'X_train', 'X_val', 'X_test', 'y_train', 'y_val' DataFrames
        """
        data_folder = Path(data_folder)
        datasets = {}
        
        # Load feature datasets
        for split in ['train', 'val', 'test']:
            x_path = data_folder / f"X_{split}.csv"
            if x_path.exists():
                datasets[f'X_{split}'] = pd.read_csv(x_path)
                logger.info(f"Loaded X_{split} with shape {datasets[f'X_{split}'].shape}")
        
        # Load target datasets
        for split in ['train', 'val']:
            y_path = data_folder / f"y_{split}.csv"
            if y_path.exists():
                datasets[f'y_{split}'] = pd.read_csv(y_path)
                logger.info(f"Loaded y_{split} with shape {datasets[f'y_{split}'].shape}")
        
        return datasets
    
    def prepare_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                        X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare features by removing non-feature columns and setting up indices.
        
        Args:
            X_train: Training features DataFrame
            X_val: Validation features DataFrame  
            X_test: Test features DataFrame
            
        Returns:
            Tuple of prepared (X_train, X_val, X_test) DataFrames
        """
        # Remove non-feature columns
        X_train_clean = X_train.drop(columns=['player_id', 'year'], errors='ignore')
        X_val_clean = X_val.drop(columns=['player_id', 'year'], errors='ignore')
        
        # Set player_id as index for test set
        X_test_clean = X_test.copy()
        if 'player_id' in X_test_clean.columns:
            X_test_clean.set_index('player_id', inplace=True)
        X_test_clean = X_test_clean.drop(columns=['year'], errors='ignore')
        
        logger.info(f"Prepared features - Train: {X_train_clean.shape}, Val: {X_val_clean.shape}, Test: {X_test_clean.shape}")
        
        return X_train_clean, X_val_clean, X_test_clean
    
    def check_data_quality(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, dict]:
        """
        Perform basic data quality checks on loaded datasets.
        
        Args:
            datasets: Dictionary of dataset names to DataFrames
            
        Returns:
            Dictionary with quality metrics for each dataset
        """
        quality_report = {}
        
        for name, df in datasets.items():
            report = {
                'shape': df.shape,
                'missing_values': df.isnull().sum().sum(),
                'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
                'duplicate_rows': df.duplicated().sum(),
                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(df.select_dtypes(include=['object']).columns)
            }
            
            if 'player_id' in df.columns:
                report['unique_players'] = df['player_id'].nunique()
                report['duplicate_players'] = df['player_id'].duplicated().sum()
            
            if 'drafted' in df.columns:
                report['draft_rate'] = df['drafted'].mean() * 100
                report['draft_counts'] = df['drafted'].value_counts().to_dict()
            
            quality_report[name] = report
        
        return quality_report
