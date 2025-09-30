"""
Model evaluation utilities for NBA draft prediction.
"""

from typing import Optional, Tuple, Union

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import auc, roc_auc_score, roc_curve

from .utils import ensure_numeric, prepare_target_variable


class ModelEvaluator:
    """
    Model evaluator for NBA draft prediction models.
    """
    
    def __init__(self):
        """Initialize ModelEvaluator."""
        logger.info("ModelEvaluator initialized")
    
    def calculate_auc_score(self, y_true: Union[pd.Series, np.ndarray],
                           y_pred_proba: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate AUC score for binary classification.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities for positive class
            
        Returns:
            AUC score
        """
        y_true_series = prepare_target_variable(y_true)
        y_pred_proba_arr = ensure_numeric(y_pred_proba)
        
        auc_score = roc_auc_score(y_true_series, y_pred_proba_arr)
        logger.info(f"AUC Score: {auc_score:.4f}")
        
        return auc_score
    
    def plot_roc_curve(self, y_true: Union[pd.Series, np.ndarray],
                      y_pred_proba: Union[pd.Series, np.ndarray],
                      title: str = "ROC Curve",
                      save_path: Optional[str] = None) -> Tuple[float, float, np.ndarray]:
        """
        Plot ROC curve and calculate AUC.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities for positive class
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Tuple of (fpr, tpr, thresholds) arrays
        """
        y_true_series = prepare_target_variable(y_true)
        y_pred_proba_arr = ensure_numeric(y_pred_proba)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true_series, y_pred_proba_arr)
        roc_auc = auc(fpr, tpr)
        
        # Create plot
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Cannot plot ROC curve.")
            return fpr, tpr, thresholds
            
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()
        
        return fpr, tpr, thresholds
    
    def analyze_predictions(self, y_true: Union[pd.Series, np.ndarray],
                          y_pred: Union[pd.Series, np.ndarray],
                          y_pred_proba: Union[pd.Series, np.ndarray]) -> dict:
        """
        Analyze model predictions and provide detailed metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary with analysis results
        """
        y_true_series = prepare_target_variable(y_true)
        y_pred_arr = ensure_numeric(y_pred)
        y_pred_proba_arr = ensure_numeric(y_pred_proba)
        
        # Basic statistics
        unique_preds, counts = np.unique(y_pred_arr, return_counts=True)
        pred_distribution = dict(zip(unique_preds, counts))
        
        # Calculate AUC
        auc_score = roc_auc_score(y_true_series, y_pred_proba_arr)
        
        # Analyze probability distribution
        prob_stats = {
            'mean': np.mean(y_pred_proba_arr),
            'std': np.std(y_pred_proba_arr),
            'min': np.min(y_pred_proba_arr),
            'max': np.max(y_pred_proba_arr),
            'median': np.median(y_pred_proba_arr)
        }
        
        # Class distribution analysis
        class_analysis = {
            'true_positive_rate': np.mean(y_pred_proba_arr[y_true_series == 1]),
            'false_positive_rate': np.mean(y_pred_proba_arr[y_true_series == 0]),
            'prediction_distribution': pred_distribution,
            'zero_predictions_percentage': np.mean(y_pred_arr == 0) * 100
        }
        
        analysis = {
            'auc_score': auc_score,
            'probability_statistics': prob_stats,
            'class_analysis': class_analysis
        }
        
        logger.info(f"Prediction analysis - AUC: {auc_score:.4f}, "
                   f"Zero predictions: {class_analysis['zero_predictions_percentage']:.1f}%")
        
        return analysis
    
    def plot_probability_distribution(self, y_true: Union[pd.Series, np.ndarray],
                                    y_pred_proba: Union[pd.Series, np.ndarray],
                                    title: str = "Probability Distribution by Class",
                                    save_path: Optional[str] = None) -> None:
        """
        Plot probability distribution for each class.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            title: Plot title
            save_path: Optional path to save the plot
        """
        y_true_series = prepare_target_variable(y_true)
        y_pred_proba_arr = ensure_numeric(y_pred_proba)
        
        # Create subplots
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Cannot plot probability distribution.")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(y_pred_proba_arr[y_true_series == 0], bins=50, alpha=0.7, 
                label='Not Drafted', color='red', density=True)
        ax1.hist(y_pred_proba_arr[y_true_series == 1], bins=50, alpha=0.7, 
                label='Drafted', color='blue', density=True)
        ax1.set_xlabel('Predicted Probability')
        ax1.set_ylabel('Density')
        ax1.set_title('Probability Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        data_to_plot = [
            y_pred_proba_arr[y_true_series == 0],
            y_pred_proba_arr[y_true_series == 1]
        ]
        labels = ['Not Drafted', 'Drafted']
        
        ax2.boxplot(data_to_plot, labels=labels)
        ax2.set_ylabel('Predicted Probability')
        ax2.set_title('Probability Distribution by Class')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Probability distribution plot saved to {save_path}")
        
        plt.show()
    
    def compare_models(self, model_results: dict) -> pd.DataFrame:
        """
        Compare multiple models based on their performance metrics.
        
        Args:
            model_results: Dictionary with model names as keys and metrics as values
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for model_name, metrics in model_results.items():
            row = {'model_name': model_name}
            row.update(metrics)
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by AUC score (descending)
        if 'auc_score' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('auc_score', ascending=False)
        
        logger.info(f"Model comparison completed for {len(model_results)} models")
        
        return comparison_df
    
    def generate_evaluation_report(self, y_true: Union[pd.Series, np.ndarray],
                                 y_pred: Union[pd.Series, np.ndarray],
                                 y_pred_proba: Union[pd.Series, np.ndarray],
                                 model_name: str = "Model") -> dict:
        """
        Generate comprehensive evaluation report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model for the report
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        # Calculate AUC
        auc_score = self.calculate_auc_score(y_true, y_pred_proba)
        
        # Analyze predictions
        analysis = self.analyze_predictions(y_true, y_pred, y_pred_proba)
        
        # Create comprehensive report
        report = {
            'model_name': model_name,
            'auc_score': auc_score,
            'prediction_analysis': analysis,
            'summary': {
                'total_samples': len(y_true),
                'positive_samples': np.sum(y_true == 1),
                'negative_samples': np.sum(y_true == 0),
                'positive_rate': np.mean(y_true == 1) * 100,
                'model_performance': 'Excellent' if auc_score > 0.9 else 
                                   'Good' if auc_score > 0.8 else 
                                   'Fair' if auc_score > 0.7 else 'Poor'
            }
        }
        
        logger.info(f"Evaluation report generated for {model_name}")
        logger.info(f"Performance: {report['summary']['model_performance']} (AUC: {auc_score:.4f})")
        
        return report
    
    def save_evaluation_results(self, results: dict, filepath: str) -> None:
        """
        Save evaluation results to JSON file.
        
        Args:
            results: Evaluation results dictionary
            filepath: Path to save the results
        """
        import json
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy_types(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {filepath}")
