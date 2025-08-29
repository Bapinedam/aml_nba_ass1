"""
Prediction utilities for NBA draft prediction models.
"""

from typing import Optional, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler

from .utils import ensure_numeric, create_submission_dataframe


class Predictor:
    """
    Predictor for NBA draft prediction models.
    """
    
    def __init__(self, model: Optional[lgb.LGBMClassifier] = None, scaler: Optional[StandardScaler] = None):
        """
        Initialize Predictor.
        
        Args:
            model: Trained LightGBM model
            scaler: Fitted StandardScaler for feature scaling
        """
        self.model = model
        self.scaler = scaler
        logger.info("Predictor initialized")
    
    def set_model(self, model: lgb.LGBMClassifier) -> None:
        """
        Set the model for prediction.
        
        Args:
            model: Trained LightGBM model
        """
        self.model = model
        logger.info("Model set for prediction")
    
    def set_scaler(self, scaler: StandardScaler) -> None:
        """
        Set the scaler for feature scaling.
        
        Args:
            scaler: Fitted StandardScaler
        """
        self.scaler = scaler
        logger.info("Scaler set for prediction")
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict probabilities for the positive class.
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities for positive class
            
        Raises:
            ValueError: If model is not set
        """
        if self.model is None:
            raise ValueError("Model not set. Use set_model() first.")
        
        # Scale features if scaler is available
        if self.scaler is not None:
            X_scaled = self.scaler.transform(ensure_numeric(X))
            logger.info("Features scaled using provided scaler")
        else:
            X_scaled = ensure_numeric(X)
            logger.info("No scaler provided, using raw features")
        
        # Make predictions
        predictions = self.model.predict_proba(X_scaled)[:, 1]
        logger.info(f"Generated predictions for {len(predictions)} samples")
        
        return predictions
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray], threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary labels.
        
        Args:
            X: Input features
            threshold: Classification threshold
            
        Returns:
            Predicted binary labels
        """
        probabilities = self.predict_proba(X)
        predictions = (probabilities >= threshold).astype(int)
        
        logger.info(f"Generated binary predictions with threshold {threshold}")
        logger.info(f"Prediction distribution: {np.bincount(predictions)}")
        
        return predictions
    
    def predict_with_confidence(self, X: Union[pd.DataFrame, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict probabilities and confidence scores.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (probabilities, confidence_scores)
        """
        probabilities = self.predict_proba(X)
        
        # Calculate confidence as distance from 0.5
        confidence = np.abs(probabilities - 0.5) * 2
        
        logger.info(f"Generated predictions with confidence scores")
        logger.info(f"Average confidence: {np.mean(confidence):.3f}")
        
        return probabilities, confidence
    
    def create_submission(self, X_test: Union[pd.DataFrame, np.ndarray],
                         player_ids: Union[pd.Index, list, np.ndarray],
                         output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Create submission DataFrame with predictions.
        
        Args:
            X_test: Test features
            player_ids: Player IDs corresponding to test samples
            output_path: Optional path to save the submission file
            
        Returns:
            Submission DataFrame with 'player_id' and 'drafted' columns
        """
        # Get predictions
        predictions = self.predict_proba(X_test)
        
        # Create submission DataFrame
        submission_df = create_submission_dataframe(predictions, player_ids, output_path)
        
        logger.info(f"Submission created with {len(submission_df)} predictions")
        
        return submission_df
    
    def analyze_predictions(self, predictions: np.ndarray) -> dict:
        """
        Analyze prediction distribution and statistics.
        
        Args:
            predictions: Model predictions (probabilities)
            
        Returns:
            Dictionary with prediction analysis
        """
        analysis = {
            'total_predictions': len(predictions),
            'mean_probability': np.mean(predictions),
            'std_probability': np.std(predictions),
            'min_probability': np.min(predictions),
            'max_probability': np.max(predictions),
            'median_probability': np.median(predictions),
            'high_confidence_predictions': np.sum(predictions > 0.8),
            'low_confidence_predictions': np.sum(predictions < 0.2),
            'uncertain_predictions': np.sum((predictions >= 0.2) & (predictions <= 0.8))
        }
        
        # Calculate percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            analysis[f'percentile_{p}'] = np.percentile(predictions, p)
        
        logger.info(f"Prediction analysis completed")
        logger.info(f"Mean probability: {analysis['mean_probability']:.3f}")
        logger.info(f"High confidence predictions: {analysis['high_confidence_predictions']}")
        
        return analysis
    
    def get_top_predictions(self, X: Union[pd.DataFrame, np.ndarray],
                          player_ids: Union[pd.Index, list, np.ndarray],
                          top_k: int = 10) -> pd.DataFrame:
        """
        Get top k predictions with highest probabilities.
        
        Args:
            X: Input features
            player_ids: Player IDs
            top_k: Number of top predictions to return
            
        Returns:
            DataFrame with top k predictions
        """
        predictions = self.predict_proba(X)
        
        # Create DataFrame with player IDs and predictions
        results_df = pd.DataFrame({
            'player_id': player_ids,
            'draft_probability': predictions
        })
        
        # Sort by probability and get top k
        top_predictions = results_df.nlargest(top_k, 'draft_probability')
        
        logger.info(f"Top {top_k} predictions retrieved")
        logger.info(f"Highest probability: {top_predictions['draft_probability'].iloc[0]:.4f}")
        
        return top_predictions
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance from the trained model.
        
        Returns:
            DataFrame with feature importance or None if not available
        """
        if self.model is None:
            logger.warning("No model available for feature importance")
            return None
        
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return None
        
        # Get feature names if available
        feature_names = None
        if hasattr(self.model, 'feature_name_'):
            feature_names = self.model.feature_name_
        elif hasattr(self.model, '_Booster'):
            feature_names = self.model._Booster.feature_name()
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names if feature_names else [f'feature_{i}' for i in range(len(self.model.feature_importances_))],
            'importance': self.model.feature_importances_
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        logger.info(f"Feature importance retrieved for {len(importance_df)} features")
        
        return importance_df
    
    def save_predictions(self, predictions: np.ndarray, filepath: str) -> None:
        """
        Save predictions to file.
        
        Args:
            predictions: Model predictions
            filepath: Path to save the predictions
        """
        np.save(filepath, predictions)
        logger.info(f"Predictions saved to {filepath}")
    
    def load_predictions(self, filepath: str) -> np.ndarray:
        """
        Load predictions from file.
        
        Args:
            filepath: Path to the saved predictions
            
        Returns:
            Loaded predictions
        """
        predictions = np.load(filepath)
        logger.info(f"Predictions loaded from {filepath}")
        return predictions
    
    def validate_predictions(self, predictions: np.ndarray) -> bool:
        """
        Validate that predictions are within expected range.
        
        Args:
            predictions: Model predictions
            
        Returns:
            True if predictions are valid, False otherwise
        """
        # Check if all predictions are between 0 and 1
        if np.any(predictions < 0) or np.any(predictions > 1):
            logger.error("Predictions contain values outside [0, 1] range")
            return False
        
        # Check for NaN values
        if np.any(np.isnan(predictions)):
            logger.error("Predictions contain NaN values")
            return False
        
        # Check for infinite values
        if np.any(np.isinf(predictions)):
            logger.error("Predictions contain infinite values")
            return False
        
        logger.info("Predictions validation passed")
        return True
