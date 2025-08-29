"""
Model training utilities for NBA draft prediction using LightGBM.
"""

from typing import Dict, List, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from .utils import (
    calculate_scale_pos_weight,
    ensure_numeric,
    log_model_info,
    prepare_target_variable,
    validate_data_shapes,
)


class ModelTrainer:
    """
    Model trainer for NBA draft prediction using LightGBM.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize ModelTrainer.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_params = None
        self.feature_names = None
        logger.info(f"ModelTrainer initialized with random_state={random_state}")
    
    def get_default_lightgbm_params(self, scale_pos_weight: float) -> Dict:
        """
        Get default LightGBM parameters.
        
        Args:
            scale_pos_weight: Weight for positive class to handle imbalance
            
        Returns:
            Dictionary of default parameters
        """
        return {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'scale_pos_weight': scale_pos_weight,
            'verbose': -1,
            'random_state': self.random_state,
            'n_estimators': 300
        }
    
    def get_hyperparameter_grid(self) -> Dict:
        """
        Get hyperparameter grid for tuning.
        
        Returns:
            Dictionary of hyperparameter ranges
        """
        return {
            'learning_rate': [0.001, 0.005, 0.01, 0.05],
            'num_leaves': [32, 64, 128, 256],
            'max_depth': [4, 6, 8, -1],
            'min_data_in_leaf': [10, 20, 50],
            'feature_fraction': [0.6, 0.8, 1.0],
            'bagging_fraction': [0.6, 0.8, 1.0],
            'bagging_freq': [3, 5, 7]
        }
    
    def scale_features(self, X_train: Union[pd.DataFrame, np.ndarray],
                      X_val: Union[pd.DataFrame, np.ndarray],
                      X_test: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Scale features using StandardScaler.
        
        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            
        Returns:
            Tuple of scaled (X_train, X_val, X_test)
        """
        # Store feature names if available
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        
        # Convert to numpy arrays
        X_train_arr = ensure_numeric(X_train)
        X_val_arr = ensure_numeric(X_val)
        X_test_arr = ensure_numeric(X_test)
        
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train_arr)
        X_val_scaled = self.scaler.transform(X_val_arr)
        X_test_scaled = self.scaler.transform(X_test_arr)
        
        logger.info("Features scaled using StandardScaler")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def train_with_grid_search(self, X_train: Union[pd.DataFrame, np.ndarray],
                              X_val: Union[pd.DataFrame, np.ndarray],
                              y_train: Union[pd.Series, np.ndarray],
                              y_val: Union[pd.Series, np.ndarray],
                              param_grid: Optional[Dict] = None,
                              cv: int = 5) -> Tuple[lgb.LGBMClassifier, GridSearchCV, pd.DataFrame]:
        """
        Train LightGBM model with grid search hyperparameter tuning.
        
        Args:
            X_train: Training features
            X_val: Validation features
            y_train: Training targets
            y_val: Validation targets
            param_grid: Hyperparameter grid (uses default if None)
            cv: Number of cross-validation folds
            
        Returns:
            Tuple of (best_model, grid_search, top_10_summary)
        """
        # Prepare data
        y_train_series = prepare_target_variable(y_train)
        y_val_series = prepare_target_variable(y_val)
        
        # Validate data shapes
        validate_data_shapes(X_train, X_val, X_train, y_train_series, y_val_series)
        
        # Scale features
        X_train_scaled, X_val_scaled, _ = self.scale_features(X_train, X_val, X_train)
        
        # Calculate scale_pos_weight
        scale_pos_weight = calculate_scale_pos_weight(y_train_series)
        
        # Get base parameters
        base_params = self.get_default_lightgbm_params(scale_pos_weight)
        
        # Use default grid if none provided
        if param_grid is None:
            param_grid = self.get_hyperparameter_grid()
        
        # Create base model
        base_model = lgb.LGBMClassifier(**base_params)
        
        # Create custom scorer for AUC
        auc_scorer = make_scorer(roc_auc_score, needs_proba=True)
        
        # Perform grid search
        logger.info("Starting grid search hyperparameter tuning...")
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=auc_scorer,
            cv=cv,
            n_jobs=-1,
            #verbose=2,
            #return_train_score=True
        )
        
        # Fit grid search
        grid_search.fit(
            X_train_scaled, y_train_series,
            eval_set=[(X_train_scaled, y_train_series), (X_val_scaled, y_val_series)],
            eval_names=['train', 'valid'],
            eval_metric='auc',
            #callbacks=[lgb.early_stopping(stopping_rounds=30)]
        )
        
        # Store best model and parameters
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Get top 10 results
        cv_results = pd.DataFrame(grid_search.cv_results_)
        cv_results_sorted = cv_results.sort_values('rank_test_score')
        top_10_summary = cv_results_sorted.head(10)
        
        # Log model information
        log_model_info(self.best_model, "Best LightGBM")
        
        return self.best_model, grid_search, top_10_summary
    
    def train_simple_model(self, X_train: Union[pd.DataFrame, np.ndarray],
                          X_val: Union[pd.DataFrame, np.ndarray],
                          y_train: Union[pd.Series, np.ndarray],
                          y_val: Union[pd.Series, np.ndarray],
                          custom_params: Optional[Dict] = None) -> lgb.LGBMClassifier:
        """
        Train LightGBM model with simple parameters (no grid search).
        
        Args:
            X_train: Training features
            X_val: Validation features
            y_train: Training targets
            y_val: Validation targets
            custom_params: Custom parameters to override defaults
            
        Returns:
            Trained LightGBM model
        """
        # Prepare data
        y_train_series = prepare_target_variable(y_train)
        y_val_series = prepare_target_variable(y_val)
        
        # Scale features
        X_train_scaled, X_val_scaled, _ = self.scale_features(X_train, X_val, X_train)
        
        # Calculate scale_pos_weight
        scale_pos_weight = calculate_scale_pos_weight(y_train_series)
        
        # Get base parameters
        params = self.get_default_lightgbm_params(scale_pos_weight)
        
        # Override with custom parameters if provided
        if custom_params:
            params.update(custom_params)
        
        # Create and train model
        logger.info("Training LightGBM model with simple parameters...")
        model = lgb.LGBMClassifier(**params)
        
        model.fit(
            X_train_scaled, y_train_series,
            eval_set=[(X_train_scaled, y_train_series), (X_val_scaled, y_val_series)],
            eval_names=['train', 'valid'],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(stopping_rounds=30)]
        )
        
        self.best_model = model
        self.best_params = params
        
        logger.info("Model training completed")
        log_model_info(model, "LightGBM")
        
        return model
    
    def get_top_models_from_grid_search(self, grid_search: GridSearchCV,
                                       X_train: Union[pd.DataFrame, np.ndarray],
                                       X_val: Union[pd.DataFrame, np.ndarray],
                                       y_train: Union[pd.Series, np.ndarray],
                                       y_val: Union[pd.Series, np.ndarray],
                                       top_k: int = 10) -> pd.DataFrame:
        """
        Get top k models from grid search results and evaluate them on validation set.
        
        Args:
            grid_search: Fitted GridSearchCV object
            X_train: Training features
            X_val: Validation features
            y_train: Training targets
            y_val: Validation targets
            top_k: Number of top models to evaluate
            
        Returns:
            DataFrame with top k models and their validation results
        """
        # Prepare data
        y_train_series = prepare_target_variable(y_train)
        y_val_series = prepare_target_variable(y_val)
        
        # Scale features
        X_train_scaled, X_val_scaled, _ = self.scale_features(X_train, X_val, X_train)
        
        # Get cv_results as DataFrame, sorted by rank_test_score
        cv_results = pd.DataFrame(grid_search.cv_results_)
        cv_results_sorted = cv_results.sort_values('rank_test_score')
        
        # Get top k models
        top_models = cv_results_sorted.head(top_k).copy()
        
        # Evaluate each model on validation set
        zero_percentages = []
        val_scores = []
        
        for idx, row in top_models.iterrows():
            # Extract parameters for this run
            params = {k.replace('param_', ''): row[k] for k in top_models.columns if k.startswith('param_')}
            
            # Add base parameters that are not in param_grid
            base_params = self.get_default_lightgbm_params(calculate_scale_pos_weight(y_train_series))
            for k, v in base_params.items():
                if k not in params:
                    params[k] = v
            
            # Fit model
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train_scaled, y_train_series,
                eval_set=[(X_train_scaled, y_train_series), (X_val_scaled, y_val_series)],
                eval_names=['train', 'valid'],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(stopping_rounds=30)]
            )
            
            # Predict on validation set
            val_pred = model.predict(X_val_scaled)
            percent_zeros = np.mean(val_pred == 0) * 100
            zero_percentages.append(percent_zeros)
            
            # Calculate AUC
            val_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            auc_val = roc_auc_score(y_val_series, val_pred_proba)
            val_scores.append(auc_val)
        
        # Add results to DataFrame
        top_models = top_models.reset_index(drop=True)
        top_models['val_auc'] = val_scores
        top_models['percent_zeros_on_val'] = zero_percentages
        
        # Keep only relevant columns
        param_cols = [c for c in top_models.columns if c.startswith('param_')]
        summary_cols = param_cols + ['mean_test_score', 'val_auc', 'percent_zeros_on_val']
        top_models_summary = top_models[summary_cols]
        
        logger.info(f"Evaluated top {top_k} models from grid search")
        
        return top_models_summary
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        
        import joblib
        joblib.dump(self.best_model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> lgb.LGBMClassifier:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded LightGBM model
        """
        import joblib
        self.best_model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return self.best_model
