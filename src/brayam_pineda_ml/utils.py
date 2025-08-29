"""
Utility functions for NBA draft prediction.
"""

from typing import Union

import numpy as np
import pandas as pd
from loguru import logger


def ensure_numeric(X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """
    Ensure that input data is numeric and convert to numpy array.
    
    Args:
        X: Input data as DataFrame or numpy array
        
    Returns:
        Numeric numpy array
        
    Raises:
        ValueError: If non-numeric columns are found
    """
    if isinstance(X, pd.DataFrame):
        bad = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if bad:
            raise ValueError(f"Non-numeric columns found in X: {bad}")
        return X.values
    return X


def calculate_scale_pos_weight(y: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate scale_pos_weight for imbalanced datasets.
    
    Args:
        y: Target variable (0 for negative class, 1 for positive class)
        
    Returns:
        Scale weight for positive class
    """
    if isinstance(y, pd.Series):
        y = y.values
    
    negative_count = (y == 0).sum()
    positive_count = (y == 1).sum()
    
    if positive_count == 0:
        logger.warning("No positive samples found in target variable")
        return 1.0
    
    scale_weight = float(negative_count / positive_count)
    logger.info(f"Calculated scale_pos_weight: {scale_weight:.2f} (negative: {negative_count}, positive: {positive_count})")
    
    return scale_weight


def prepare_target_variable(y: Union[pd.DataFrame, pd.Series, np.ndarray]) -> pd.Series:
    """
    Prepare target variable by ensuring it's a pandas Series.
    
    Args:
        y: Target variable in various formats
        
    Returns:
        Target variable as pandas Series
    """
    if isinstance(y, pd.DataFrame):
        y = y.squeeze("columns")  # Convert to Series
    elif isinstance(y, np.ndarray):
        y = pd.Series(y)
    
    return y


def validate_data_shapes(X_train: Union[pd.DataFrame, np.ndarray], 
                        X_val: Union[pd.DataFrame, np.ndarray],
                        X_test: Union[pd.DataFrame, np.ndarray],
                        y_train: Union[pd.Series, np.ndarray],
                        y_val: Union[pd.Series, np.ndarray]) -> None:
    """
    Validate that data shapes are consistent.
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        y_train: Training targets
        y_val: Validation targets
        
    Raises:
        ValueError: If shapes are inconsistent
    """
    # Convert to numpy arrays for shape checking
    X_train_arr = ensure_numeric(X_train)
    X_val_arr = ensure_numeric(X_val)
    X_test_arr = ensure_numeric(X_test)
    
    # Check feature dimensions
    if X_train_arr.shape[1] != X_val_arr.shape[1]:
        raise ValueError(f"Feature dimension mismatch: X_train has {X_train_arr.shape[1]} features, X_val has {X_val_arr.shape[1]}")
    
    if X_train_arr.shape[1] != X_test_arr.shape[1]:
        raise ValueError(f"Feature dimension mismatch: X_train has {X_train_arr.shape[1]} features, X_test has {X_test_arr.shape[1]}")
    
    # Check target dimensions
    y_train_series = prepare_target_variable(y_train)
    y_val_series = prepare_target_variable(y_val)
    
    if X_train_arr.shape[0] != len(y_train_series):
        raise ValueError(f"Sample count mismatch: X_train has {X_train_arr.shape[0]} samples, y_train has {len(y_train_series)}")
    
    if X_val_arr.shape[0] != len(y_val_series):
        raise ValueError(f"Sample count mismatch: X_val has {X_val_arr.shape[0]} samples, y_val has {len(y_val_series)}")
    
    logger.info("Data shape validation passed")


def get_feature_names(X: Union[pd.DataFrame, np.ndarray]) -> list:
    """
    Get feature names from input data.
    
    Args:
        X: Input features
        
    Returns:
        List of feature names or None if not available
    """
    if isinstance(X, pd.DataFrame):
        return X.columns.tolist()
    return None


def log_model_info(model, model_name: str = "Model") -> None:
    """
    Log basic information about a trained model.
    
    Args:
        model: Trained model object
        model_name: Name of the model for logging
    """
    logger.info(f"{model_name} type: {type(model).__name__}")
    
    # Log model parameters if available
    if hasattr(model, 'get_params'):
        params = model.get_params()
        logger.info(f"{model_name} parameters: {params}")
    
    # Log feature importance if available
    if hasattr(model, 'feature_importances_'):
        n_features = len(model.feature_importances_)
        top_features = np.argsort(model.feature_importances_)[-5:][::-1]
        logger.info(f"{model_name} has {n_features} features")
        logger.info(f"{model_name} top 5 feature indices: {top_features.tolist()}")


def create_submission_dataframe(predictions: np.ndarray, 
                               player_ids: Union[pd.Index, list, np.ndarray],
                               output_path: str = None) -> pd.DataFrame:
    """
    Create submission DataFrame with predictions.
    
    Args:
        predictions: Model predictions (probabilities)
        player_ids: Player IDs corresponding to predictions
        output_path: Optional path to save the submission file
        
    Returns:
        Submission DataFrame with 'player_id' and 'drafted' columns
    """
    submission_df = pd.DataFrame({
        'player_id': player_ids,
        'drafted': predictions
    })
    
    if output_path:
        submission_df.to_csv(output_path, index=False)
        logger.info(f"Submission saved to {output_path}")
    
    return submission_df
