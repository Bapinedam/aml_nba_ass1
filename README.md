# Brayam Pineda ML Package

A comprehensive machine learning package for NBA draft prediction using LightGBM. This package provides tools for data loading, preprocessing, model training, evaluation, and prediction generation.

## Features

- **Data Loading & Preprocessing**: Load and process NBA player datasets with automatic height parsing
- **Model Training**: Train LightGBM models with hyperparameter tuning using grid search
- **Model Evaluation**: Comprehensive evaluation metrics including ROC curves and AUC analysis
- **Prediction Generation**: Generate predictions and create submission files
- **Utility Functions**: Data validation, feature scaling, and model utilities

## Installation

### From TestPyPI

```bash
pip install -i https://test.pypi.org/simple/ brayam-pineda-ml
```

### From Source

```bash
git clone https://github.com/yourusername/brayam-pineda-ml.git
cd brayam-pineda-ml
pip install -e .
```

## Requirements

- Python 3.11.4+
- scikit-learn 1.5.1
- pandas 2.2.2
- numpy >= 1.24.0
- lightgbm 4.4.0
- joblib 1.4.2
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- hyperopt 0.2.7
- lime 0.2.0.1
- wandb 0.17.4
- loguru >= 0.7.0
- tqdm >= 4.65.0
- typer >= 0.9.0
- python-dotenv >= 1.0.0

## Quick Start

### Basic Usage

```python
from brayam_pineda_ml import DataLoader, ModelTrainer, ModelEvaluator, Predictor

# Load data
data_loader = DataLoader()
datasets = data_loader.load_processed_data()

# Prepare features
X_train, X_val, X_test = data_loader.prepare_features(
    datasets['X_train'], 
    datasets['X_val'], 
    datasets['X_test']
)

# Train model
trainer = ModelTrainer()
model = trainer.train_simple_model(
    X_train, X_val, 
    datasets['y_train'], datasets['y_val']
)

# Evaluate model
evaluator = ModelEvaluator()
predictions = model.predict_proba(X_val)
auc_score = evaluator.calculate_auc_score(datasets['y_val'], predictions)

# Generate predictions
predictor = Predictor(model, trainer.scaler)
submission = predictor.create_submission(X_test, X_test.index)
```

### Advanced Usage with Hyperparameter Tuning

```python
from brayam_pineda_ml import ModelTrainer

# Train with grid search
trainer = ModelTrainer()
model = trainer.train_with_grid_search(
    X_train, X_val, 
    y_train, y_val,
    cv=5
)

# Get top models from grid search
top_models = trainer.get_top_models_from_grid_search(
    grid_search, X_train, X_val, y_train, y_val, top_k=10
)
```

### Data Quality Analysis

```python
from brayam_pineda_ml import DataLoader

# Load raw datasets
data_loader = DataLoader()
raw_datasets = data_loader.load_raw_datasets()

# Check data quality
quality_report = data_loader.check_data_quality(raw_datasets)
print(quality_report)
```

### Model Evaluation

```python
from brayam_pineda_ml import ModelEvaluator

evaluator = ModelEvaluator()

# Plot ROC curve
fpr, tpr, thresholds = evaluator.plot_roc_curve(y_true, y_pred_proba)

# Analyze predictions
analysis = evaluator.analyze_predictions(y_true, y_pred, y_pred_proba)

# Generate comprehensive report
report = evaluator.generate_evaluation_report(y_true, y_pred, y_pred_proba, "My Model")
```

## API Reference

### DataLoader

The `DataLoader` class provides functionality for loading and preprocessing NBA player datasets.

#### Methods

- `load_raw_datasets()`: Load all CSV files from the raw data folder
- `load_processed_data()`: Load processed train/validation/test splits
- `prepare_features()`: Remove non-feature columns and prepare data for modeling
- `check_data_quality()`: Perform data quality checks and generate reports

### ModelTrainer

The `ModelTrainer` class handles model training and hyperparameter tuning.

#### Methods

- `train_simple_model()`: Train a LightGBM model with default or custom parameters
- `train_with_grid_search()`: Train with hyperparameter tuning using grid search
- `get_top_models_from_grid_search()`: Evaluate top models from grid search results
- `save_model()` / `load_model()`: Save and load trained models

### ModelEvaluator

The `ModelEvaluator` class provides comprehensive model evaluation tools.

#### Methods

- `calculate_auc_score()`: Calculate AUC score for binary classification
- `plot_roc_curve()`: Plot ROC curve and calculate AUC
- `analyze_predictions()`: Analyze prediction distribution and statistics
- `plot_probability_distribution()`: Plot probability distribution by class
- `compare_models()`: Compare multiple models based on performance metrics
- `generate_evaluation_report()`: Generate comprehensive evaluation reports

### Predictor

The `Predictor` class handles prediction generation and submission creation.

#### Methods

- `predict_proba()`: Predict probabilities for the positive class
- `predict()`: Predict binary labels with custom threshold
- `predict_with_confidence()`: Predict probabilities and confidence scores
- `create_submission()`: Create submission DataFrame with predictions
- `analyze_predictions()`: Analyze prediction distribution and statistics
- `get_top_predictions()`: Get top k predictions with highest probabilities
- `get_feature_importance()`: Get feature importance from trained model

### Utility Functions

- `parse_height_to_cm()`: Convert height values to centimeters
- `ensure_numeric()`: Ensure input data is numeric
- `calculate_scale_pos_weight()`: Calculate scale_pos_weight for imbalanced datasets
- `prepare_target_variable()`: Prepare target variable for modeling
- `validate_data_shapes()`: Validate data shape consistency
- `create_submission_dataframe()`: Create submission DataFrame

## Height Parsing

The package includes a sophisticated height parsing function that supports multiple formats:

### Supported Formats

1. **Feet/Inches Format**: `6'11''`, `5'9''`, `7'0''`
2. **Date Format**: `1-Jun` → `6'1''`, `11-May` → `5'11''`

### Usage

```python
from brayam_pineda_ml import parse_height_to_cm

# Parse different height formats
height1 = parse_height_to_cm("6'11''")  # Returns 210.82 cm
height2 = parse_height_to_cm("1-Jun")   # Returns 185.42 cm (6'1")
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=brayam_pineda_ml

# Run specific test file
pytest tests/test_data_loader.py
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

### Building and Publishing

```bash
# Build package
python -m build

# Upload to TestPyPI
twine upload --repository testpypi dist/*
```

## Project Structure

```
brayam-pineda-ml/
├── src/
│   └── brayam_pineda_ml/
│       ├── __init__.py
│       ├── data_loader.py
│       ├── model_trainer.py
│       ├── evaluator.py
│       ├── predictor.py
│       └── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_utils.py
│   └── ...
├── pyproject.toml
├── README.md
└── ...
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Brayam Alexander Pineda**
- Email: brayam.pineda@student.uts.edu.au
- Student ID: 25587799
- Group ID: 26

## Acknowledgments

This package was developed as part of the Advanced Machine Learning course at the University of Technology Sydney. Special thanks to the course instructors and teaching assistants for their guidance and support.

