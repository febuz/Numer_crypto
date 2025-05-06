# Sample Model Comparison for Numerai Crypto

## Model Performance Metrics

| Model Name | Category | Train RMSE | Val RMSE | Std Val RMSE | Overfitting Ratio |
|------------|----------|------------|----------|--------------|-------------------|
| H2O_StackedEnsemble | h2o | 0.1534 | 0.1672 | 0.0085 | 0.9174 |
| XGBoost | sklearn | 0.1595 | 0.1698 | 0.0093 | 0.9393 |
| LightGBM | sklearn | 0.1582 | 0.1712 | 0.0102 | 0.9241 |
| GPU_Boosting_Ensemble | custom | 0.1548 | 0.1725 | 0.0089 | 0.8974 |
| CatBoost | sklearn | 0.1602 | 0.1743 | 0.0112 | 0.9192 |
| H2O_GBM | h2o | 0.1621 | 0.1762 | 0.0096 | 0.9200 |
| H2O_XGBoost | h2o | 0.1635 | 0.1780 | 0.0103 | 0.9186 |
| H2O_TPOT_Ensemble | custom | 0.1602 | 0.1795 | 0.0110 | 0.8924 |
| RandomForest | sklearn | 0.1685 | 0.1842 | 0.0125 | 0.9148 |
| H2O_DeepLearning | h2o | 0.1698 | 0.1864 | 0.0132 | 0.9109 |
| H2O_GLM | h2o | 0.1963 | 0.2015 | 0.0098 | 0.9742 |
| BayesianRidge | sklearn | 0.1985 | 0.2042 | 0.0087 | 0.9721 |
| ExtraTrees | sklearn | 0.1784 | 0.2065 | 0.0142 | 0.8639 |
| H2O_DRF | h2o | 0.1823 | 0.2076 | 0.0135 | 0.8781 |
| Ridge | sklearn | 0.2012 | 0.2083 | 0.0092 | 0.9659 |
| ElasticNet | sklearn | 0.2045 | 0.2124 | 0.0095 | 0.9628 |
| Lasso | sklearn | 0.2087 | 0.2153 | 0.0102 | 0.9694 |
| SGDRegressor | sklearn | 0.2102 | 0.2175 | 0.0108 | 0.9664 |
| HighMemCrypto | custom | 0.1642 | 0.1892 | 0.0128 | 0.8679 |
| MLP | sklearn | 0.1892 | 0.2234 | 0.0143 | 0.8469 |
| DecisionTree | sklearn | 0.1325 | 0.2375 | 0.0185 | 0.5579 |
| Enhanced_Ensemble | ensemble | 0.1512 | 0.1654 | 0.0072 | 0.9141 |
| Enhanced_Unique | ensemble | 0.1512 | 0.1664 | 0.0075 | 0.9087 |

## Enhanced Ensemble Details

The enhanced ensemble combines the following models:
- H2O_StackedEnsemble (weight: 39.8%)
- XGBoost (weight: 37.5%)
- LightGBM (weight: 22.7%)

## Model Runtime Analysis

| Category | Avg. Train Time (s) | Models Trained | Parallel Training |
|----------|---------------------|----------------|-------------------|
| H2O AutoML | 1800 | 42 | Yes |
| Sklearn Models | 582 | 12 | Yes |
| Custom Models | 745 | 3 | Yes |
| Ensemble Creation | 12 | 1 | N/A |

## Feature Importance Analysis

Top 10 features across models:
1. pvm_0042 (importance: 0.0832)
2. pvm_0124 (importance: 0.0756)
3. pvm_0033 (importance: 0.0721)
4. sentiment_0015 (importance: 0.0687)
5. onchain_0215 (importance: 0.0653)
6. pvm_0076 (importance: 0.0612)
7. sentiment_0042 (importance: 0.0598)
8. onchain_0033 (importance: 0.0572)
9. pvm_0008 (importance: 0.0548)
10. sentiment_0304 (importance: 0.0536)

## Prediction Uniqueness Analysis

The standard and unique ensemble predictions show:
- Correlation: 0.9842
- Average deviation: 0.0158
- Maximum deviation: 0.0476

The unique model achieves a higher weight for volatile assets and de-emphasizes momentum features compared to the meta-model approach.

## Expected Live Performance

Based on cross-validation results and historical model performance:
- Expected RMSE range: 0.1654-0.1754
- Expected correlation with meta-model: 0.75-0.85
- Expected percentile ranking: Top 10-15%