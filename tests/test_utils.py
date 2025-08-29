"""
Unit tests for utils module.
"""

import pytest
import numpy as np
import pandas as pd

from brayam_pineda_ml.utils import (
    ensure_numeric,
    calculate_scale_pos_weight,
    prepare_target_variable,
    validate_data_shapes,
    get_feature_names,
    log_model_info,
    create_submission_dataframe
)


class TestEnsureNumeric:
    """Test cases for ensure_numeric function."""
    
    def test_numpy_array(self):
        """Test with numpy array."""
        arr = np.array([[1, 2], [3, 4]])
        result = ensure_numeric(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)
    
    def test_pandas_dataframe_numeric(self):
        """Test with numeric pandas DataFrame."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        result = ensure_numeric(df)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, df.values)
    
    def test_pandas_dataframe_non_numeric(self):
        """Test with non-numeric pandas DataFrame."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        with pytest.raises(ValueError, match="Non-numeric columns found"):
            ensure_numeric(df)


class TestCalculateScalePosWeight:
    """Test cases for calculate_scale_pos_weight function."""
    
    def test_balanced_dataset(self):
        """Test with balanced dataset."""
        y = np.array([0, 1, 0, 1])
        weight = calculate_scale_pos_weight(y)
        assert weight == 1.0
    
    def test_imbalanced_dataset(self):
        """Test with imbalanced dataset."""
        y = np.array([0, 0, 0, 0, 1])  # 4 negative, 1 positive
        weight = calculate_scale_pos_weight(y)
        assert weight == 4.0
    
    def test_no_positive_samples(self):
        """Test with no positive samples."""
        y = np.array([0, 0, 0])
        weight = calculate_scale_pos_weight(y)
        assert weight == 1.0
    
    def test_pandas_series(self):
        """Test with pandas Series."""
        y = pd.Series([0, 0, 1, 1, 1])
        weight = calculate_scale_pos_weight(y)
        assert weight == 2.0 / 3.0


class TestPrepareTargetVariable:
    """Test cases for prepare_target_variable function."""
    
    def test_pandas_dataframe(self):
        """Test with pandas DataFrame."""
        df = pd.DataFrame({'target': [0, 1, 0]})
        result = prepare_target_variable(df)
        assert isinstance(result, pd.Series)
        assert result.name == 'target'
    
    def test_pandas_series(self):
        """Test with pandas Series."""
        series = pd.Series([0, 1, 0])
        result = prepare_target_variable(series)
        assert isinstance(result, pd.Series)
        pd.testing.assert_series_equal(result, series)
    
    def test_numpy_array(self):
        """Test with numpy array."""
        arr = np.array([0, 1, 0])
        result = prepare_target_variable(arr)
        assert isinstance(result, pd.Series)
        np.testing.assert_array_equal(result.values, arr)


class TestValidateDataShapes:
    """Test cases for validate_data_shapes function."""
    
    def test_valid_shapes(self):
        """Test with valid data shapes."""
        X_train = np.random.rand(100, 10)
        X_val = np.random.rand(25, 10)
        X_test = np.random.rand(50, 10)
        y_train = np.random.randint(0, 2, 100)
        y_val = np.random.randint(0, 2, 25)
        
        # Should not raise any exception
        validate_data_shapes(X_train, X_val, X_test, y_train, y_val)
    
    def test_feature_dimension_mismatch(self):
        """Test with feature dimension mismatch."""
        X_train = np.random.rand(100, 10)
        X_val = np.random.rand(25, 5)  # Different number of features
        X_test = np.random.rand(50, 10)
        y_train = np.random.randint(0, 2, 100)
        y_val = np.random.randint(0, 2, 25)
        
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            validate_data_shapes(X_train, X_val, X_test, y_train, y_val)
    
    def test_sample_count_mismatch(self):
        """Test with sample count mismatch."""
        X_train = np.random.rand(100, 10)
        X_val = np.random.rand(25, 10)
        X_test = np.random.rand(50, 10)
        y_train = np.random.randint(0, 2, 100)
        y_val = np.random.randint(0, 2, 30)  # Different number of samples
        
        with pytest.raises(ValueError, match="Sample count mismatch"):
            validate_data_shapes(X_train, X_val, X_test, y_train, y_val)


class TestGetFeatureNames:
    """Test cases for get_feature_names function."""
    
    def test_pandas_dataframe(self):
        """Test with pandas DataFrame."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        names = get_feature_names(df)
        assert names == ['a', 'b']
    
    def test_numpy_array(self):
        """Test with numpy array."""
        arr = np.random.rand(10, 5)
        names = get_feature_names(arr)
        assert names is None


class TestLogModelInfo:
    """Test cases for log_model_info function."""
    
    def test_model_with_params(self):
        """Test with model that has get_params method."""
        class MockModel:
            def get_params(self):
                return {'param1': 1, 'param2': 2}
        
        model = MockModel()
        # Should not raise any exception
        log_model_info(model, "Test Model")
    
    def test_model_without_params(self):
        """Test with model that doesn't have get_params method."""
        class MockModel:
            pass
        
        model = MockModel()
        # Should not raise any exception
        log_model_info(model, "Test Model")


class TestCreateSubmissionDataframe:
    """Test cases for create_submission_dataframe function."""
    
    def test_create_submission(self):
        """Test creating submission DataFrame."""
        predictions = np.array([0.1, 0.5, 0.9])
        player_ids = ['player1', 'player2', 'player3']
        
        submission_df = create_submission_dataframe(predictions, player_ids)
        
        assert isinstance(submission_df, pd.DataFrame)
        assert list(submission_df.columns) == ['player_id', 'drafted']
        assert len(submission_df) == 3
        np.testing.assert_array_equal(submission_df['drafted'].values, predictions)
        assert list(submission_df['player_id']) == player_ids
    
    def test_create_submission_with_output_path(self, tmp_path):
        """Test creating submission DataFrame with output path."""
        predictions = np.array([0.1, 0.5, 0.9])
        player_ids = ['player1', 'player2', 'player3']
        output_path = tmp_path / "submission.csv"
        
        submission_df = create_submission_dataframe(predictions, player_ids, str(output_path))
        
        assert output_path.exists()
        assert isinstance(submission_df, pd.DataFrame)
    
    def test_create_submission_with_pandas_index(self):
        """Test creating submission DataFrame with pandas Index."""
        predictions = np.array([0.1, 0.5, 0.9])
        player_ids = pd.Index(['player1', 'player2', 'player3'])
        
        submission_df = create_submission_dataframe(predictions, player_ids)
        
        assert isinstance(submission_df, pd.DataFrame)
        assert len(submission_df) == 3
