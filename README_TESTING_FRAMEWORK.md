# Numer_crypto Quick Testing Framework

A lightweight framework for rapidly testing new features and models without modifying the main codebase.

## Overview

The Quick Testing Framework allows you to:

1. Quickly test new feature generation ideas
2. Evaluate and compare model performance
3. Visualize results with automatic charts
4. Test with minimal resource usage
5. Integrate seamlessly with the existing codebase

## Key Features

- **Lightweight**: Uses small data samples for quick feedback cycles
- **Modular**: Easily swap in/out features and models
- **Visualization**: Automatic charts for model performance and feature importance
- **Fast Execution**: Optimized for speed (<10 minutes execution time)
- **Low Resource Usage**: Works on typical development machines

## Getting Started

### Basic Usage

```python
# Import the framework
from scripts.quick_test_framework import QuickTestFramework

# Initialize the framework
framework = QuickTestFramework(
    output_dir="./test_results",  # Where to save results
    sample_size=50000  # Use a small sample for fast testing
)

# Load a sample of data
framework.load_sample_data()

# Test a model with default features
result = framework.test_model(model_name='xgboost')
```

### Command Line Usage

You can also run the framework from the command line:

```bash
python scripts/quick_test_framework.py --model xgboost --feature-generator polars --cv-folds 3
```

## Testing Custom Features

The framework makes it easy to test your own feature ideas:

```python
def my_custom_features(df):
    """Function that adds new features to a DataFrame"""
    result = df.copy()
    
    # Add your feature logic here
    result['new_feature'] = df['column1'] * df['column2']
    
    return result

# Test your custom features
result = framework.test_custom_feature_function(
    feature_fn=my_custom_features,
    model_name='xgboost'
)
```

## Comparing Models

Compare multiple models on the same dataset:

```python
# Compare several models
comparison = framework.compare_models(
    models=['xgboost', 'lightgbm'],
    cv_folds=3
)
```

## Comparing Feature Sets

Compare different feature generation approaches:

```python
# Compare different feature generators
comparison = framework.compare_feature_sets(
    feature_generators=['polars', 'polars_gpu'],
    model_name='xgboost'
)
```

## Example Script

An example script is provided to demonstrate how to use the framework:

```bash
python scripts/test_example.py
```

This script demonstrates:
- Testing a model with default features
- Testing a custom feature function
- Comparing different models
- Testing with a feature generator

## Output

The framework saves results to the specified output directory:
- `cv_results.png`: Cross-validation results by fold
- `feature_importance.png`: Top feature importance
- `model_comparison.png`: Comparison of model performance
- `feature_set_comparison.png`: Scatter plot of feature sets vs. performance
- `feature_set_performance.png`: Bar chart of feature set performance

## Integration with Existing Codebase

The framework automatically discovers available models and feature generators in the codebase, making it seamlessly integrated with the existing code.

## Requirements

- Python 3.7+
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- Specific requirements for any models/features you want to test (e.g., xgboost, lightgbm)

## Advanced Usage

### Custom Models

You can test custom model implementations:

```python
# Create a custom model class
class MyCustomModel:
    def __init__(self, **kwargs):
        # Initialize your model
        pass
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        # Train your model
        return {'feature_importance': {}}
        
    def predict(self, X):
        # Generate predictions
        return np.zeros(len(X))
        
    def get_feature_importance(self):
        # Return feature importance
        return {'feature1': 1.0}

# Test your custom model
custom_model = MyCustomModel()
result = framework.test_model(custom_model=custom_model)
```

### Custom Feature Generators

You can also test custom feature generator implementations:

```python
# Create a custom feature generator class
class MyFeatureGenerator:
    def __init__(self, **kwargs):
        # Initialize your generator
        pass
        
    def generate_all_features(self, df, group_col, numeric_cols, date_col):
        # Generate features
        return df

# Test your custom generator
custom_generator = MyFeatureGenerator()
features_df = framework.generate_features(custom_generator=custom_generator)
```