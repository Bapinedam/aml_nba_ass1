"""
Unit tests for data_loader module.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from brayam_pineda_ml.data_loader import DataLoader, parse_height_to_cm


class TestParseHeightToCm:
    """Test cases for parse_height_to_cm function."""
    
    def test_feet_inches_format(self):
        """Test feet/inches format parsing."""
        assert parse_height_to_cm("6'11''") == pytest.approx(210.82, rel=1e-2)
        assert parse_height_to_cm("5'9''") == pytest.approx(175.26, rel=1e-2)
        assert parse_height_to_cm("7'0''") == pytest.approx(213.36, rel=1e-2)
    
    def test_date_format(self):
        """Test date format parsing."""
        assert parse_height_to_cm("1-Jun") == pytest.approx(185.42, rel=1e-2)  # 6'1"
        assert parse_height_to_cm("11-May") == pytest.approx(180.34, rel=1e-2)  # 5'11"
        assert parse_height_to_cm("3-Mar") == pytest.approx(167.64, rel=1e-2)   # 3'3"
    
    def test_invalid_formats(self):
        """Test invalid height formats."""
        assert np.isnan(parse_height_to_cm("invalid"))
        assert np.isnan(parse_height_to_cm("6'12''"))  # Invalid inches
        assert np.isnan(parse_height_to_cm("13-Jun"))  # Invalid day
        assert np.isnan(parse_height_to_cm(""))
        assert np.isnan(parse_height_to_cm(None))
        assert np.isnan(parse_height_to_cm(np.nan))
    
    def test_edge_cases(self):
        """Test edge cases."""
        assert parse_height_to_cm("0'0''") == pytest.approx(0.0, rel=1e-2)
        assert parse_height_to_cm("12'11''") == pytest.approx(393.7, rel=1e-2)


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.data_loader = DataLoader()
    
    def test_init(self):
        """Test DataLoader initialization."""
        assert self.data_loader.data_folder == self.data_loader.data_folder
    
    def test_load_raw_datasets_file_not_found(self):
        """Test loading datasets when folder doesn't exist."""
        with pytest.raises(FileNotFoundError):
            self.data_loader.load_raw_datasets()
    
    @patch('os.listdir')
    @patch('builtins.open')
    @patch('pandas.read_csv')
    def test_load_raw_datasets_success(self, mock_read_csv, mock_open, mock_listdir):
        """Test successful dataset loading."""
        # Mock file listing
        mock_listdir.return_value = ['train.csv', 'test.csv', 'metadata.csv']
        
        # Mock file reading
        mock_open.return_value.__enter__.return_value.readline.return_value = "player_id,ht,drafted\n"
        
        # Mock DataFrame creation
        mock_df = pd.DataFrame({
            'player_id': ['1', '2', '3'],
            'ht': ['6\'11\'\'', '5\'9\'\'', '1-Jun'],
            'drafted': [1, 0, 1]
        })
        mock_read_csv.return_value = mock_df
        
        # Test loading
        datasets = self.data_loader.load_raw_datasets()
        
        assert 'train' in datasets
        assert 'test' in datasets
        assert 'metadata' in datasets
        assert len(datasets) == 3
    
    def test_prepare_features(self):
        """Test feature preparation."""
        # Create test DataFrames
        X_train = pd.DataFrame({
            'player_id': ['1', '2', '3'],
            'year': [2020, 2020, 2020],
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        X_val = pd.DataFrame({
            'player_id': ['4', '5'],
            'year': [2020, 2020],
            'feature1': [7, 8],
            'feature2': [9, 10]
        })
        
        X_test = pd.DataFrame({
            'player_id': ['6', '7'],
            'year': [2020, 2020],
            'feature1': [11, 12],
            'feature2': [13, 14]
        })
        
        # Test preparation
        X_train_clean, X_val_clean, X_test_clean = self.data_loader.prepare_features(X_train, X_val, X_test)
        
        # Check that non-feature columns are removed
        assert 'player_id' not in X_train_clean.columns
        assert 'year' not in X_train_clean.columns
        assert 'player_id' not in X_val_clean.columns
        assert 'year' not in X_val_clean.columns
        
        # Check that test set has player_id as index
        assert X_test_clean.index.name == 'player_id'
        assert 'year' not in X_test_clean.columns
        
        # Check feature columns are preserved
        assert 'feature1' in X_train_clean.columns
        assert 'feature2' in X_train_clean.columns
    
    def test_check_data_quality(self):
        """Test data quality checking."""
        # Create test datasets
        datasets = {
            'train': pd.DataFrame({
                'player_id': ['1', '2', '3', '1'],  # Duplicate player_id
                'drafted': [1, 0, 1, 0],
                'feature1': [1, 2, 3, 4],
                'feature2': [5, 6, np.nan, 8]  # Missing value
            }),
            'test': pd.DataFrame({
                'player_id': ['4', '5'],
                'feature1': [9, 10],
                'feature2': [11, 12]
            })
        }
        
        # Test quality check
        quality_report = self.data_loader.check_data_quality(datasets)
        
        # Check that both datasets are in the report
        assert 'train' in quality_report
        assert 'test' in quality_report
        
        # Check train dataset metrics
        train_metrics = quality_report['train']
        assert train_metrics['shape'] == (4, 4)
        assert train_metrics['missing_values'] == 1
        assert train_metrics['duplicate_players'] == 1
        assert train_metrics['draft_rate'] == 50.0  # 2 out of 4 drafted
        
        # Check test dataset metrics
        test_metrics = quality_report['test']
        assert test_metrics['shape'] == (2, 3)
        assert test_metrics['missing_values'] == 0
        assert test_metrics['unique_players'] == 2
    
    @patch('pathlib.Path.exists')
    @patch('pandas.read_csv')
    def test_load_processed_data(self, mock_read_csv, mock_exists):
        """Test loading processed data."""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock DataFrame creation
        mock_df = pd.DataFrame({'feature1': [1, 2, 3]})
        mock_read_csv.return_value = mock_df
        
        # Test loading
        datasets = self.data_loader.load_processed_data()
        
        # Check that expected datasets are loaded
        expected_keys = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val']
        for key in expected_keys:
            assert key in datasets
