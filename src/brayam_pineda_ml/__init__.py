"""
Brayam Pineda ML Package

A machine learning package for NBA draft prediction and weather forecasting using LightGBM and XGBoost.
"""

__version__ = "0.1.0"
__author__ = "Brayam Alexander Pineda"
__email__ = "brayam.pineda@student.uts.edu.au"

from .data_loader import DataLoader, parse_height_to_cm
from .model_trainer import ModelTrainer
from .evaluator import ModelEvaluator
from .predictor import Predictor
from .utils import ensure_numeric, calculate_scale_pos_weight
from .weather_data_processor import WeatherDataProcessor

__all__ = [
    "DataLoader",
    "parse_height_to_cm", 
    "ModelTrainer",
    "ModelEvaluator",
    "Predictor",
    "ensure_numeric",
    "calculate_scale_pos_weight",
    "WeatherDataProcessor",
]
