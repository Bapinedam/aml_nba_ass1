#!/usr/bin/env python3
"""
Example usage of the Brayam Pineda ML Package for NBA draft prediction.
"""

import numpy as np
import pandas as pd
from brayam_pineda_ml import (
    DataLoader, 
    ModelTrainer, 
    ModelEvaluator, 
    Predictor,
    parse_height_to_cm
)

def main():
    """Demonstrate the package functionality."""
    
    print("=== Brayam Pineda ML Package Example ===\n")
    
    # 1. Test height parsing
    print("1. Testing height parsing:")
    test_heights = ["6'11''", "5'9''", "1-Jun", "11-May"]
    for height in test_heights:
        cm = parse_height_to_cm(height)
        print(f"   {height} -> {cm:.2f} cm")
    print()
    
    # 2. Create sample data for demonstration
    print("2. Creating sample data:")
    np.random.seed(42)
    
    # Generate sample features
    n_samples = 1000
    n_features = 10
    
    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    X_val = pd.DataFrame(
        np.random.randn(200, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    X_test = pd.DataFrame(
        np.random.randn(100, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Generate imbalanced target (similar to NBA draft data)
    y_train = np.random.choice([0, 1], size=n_samples, p=[0.992, 0.008])
    y_val = np.random.choice([0, 1], size=200, p=[0.992, 0.008])
    
    print(f"   Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"   Validation set: {X_val.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    print(f"   Draft rate: {np.mean(y_train)*100:.2f}%")
    print()
    
    # 3. Train model
    print("3. Training LightGBM model:")
    trainer = ModelTrainer(random_state=42)
    
    # Use custom parameters for faster training
    custom_params = {
        'learning_rate': 0.1,
        'num_leaves': 31,
        'max_depth': 6,
        'n_estimators': 100
    }
    
    model = trainer.train_simple_model(
        X_train, X_val, y_train, y_val, custom_params
    )
    print()
    
    # 4. Evaluate model
    print("4. Evaluating model:")
    evaluator = ModelEvaluator()
    
    # Get predictions
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    y_val_pred = model.predict(X_val)
    
    # Calculate AUC
    auc_score = evaluator.calculate_auc_score(y_val, y_val_pred_proba)
    print(f"   Validation AUC: {auc_score:.4f}")
    
    # Analyze predictions
    analysis = evaluator.analyze_predictions(y_val, y_val_pred, y_val_pred_proba)
    print(f"   Mean prediction probability: {analysis['probability_statistics']['mean']:.4f}")
    print(f"   Zero predictions percentage: {analysis['class_analysis']['zero_predictions_percentage']:.1f}%")
    print()
    
    # 5. Generate predictions
    print("5. Generating predictions:")
    predictor = Predictor(model, trainer.scaler)
    
    # Get test predictions
    test_predictions = predictor.predict_proba(X_test)
    
    # Analyze predictions
    pred_analysis = predictor.analyze_predictions(test_predictions)
    print(f"   Test predictions generated: {pred_analysis['total_predictions']}")
    print(f"   Mean probability: {pred_analysis['mean_probability']:.4f}")
    print(f"   High confidence predictions: {pred_analysis['high_confidence_predictions']}")
    
    # Get top predictions
    player_ids = [f"player_{i}" for i in range(len(X_test))]
    top_predictions = predictor.get_top_predictions(X_test, player_ids, top_k=5)
    print(f"   Top 5 predictions:")
    for _, row in top_predictions.iterrows():
        print(f"     {row['player_id']}: {row['draft_probability']:.4f}")
    print()
    
    # 6. Create submission
    print("6. Creating submission file:")
    submission = predictor.create_submission(X_test, player_ids, "example_submission.csv")
    print(f"   Submission created with {len(submission)} predictions")
    print(f"   Saved to: example_submission.csv")
    print()
    
    print("=== Example completed successfully! ===")

if __name__ == "__main__":
    main()
