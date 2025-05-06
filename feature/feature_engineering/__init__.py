"""
Feature Engineering module for Numerai Crypto

This module provides tools for generating new features from the existing data,
including polynomial features, statistical features, and domain-specific crypto features.
"""

from .polynomial_features import PolynomialFeatureGenerator

__all__ = ['PolynomialFeatureGenerator']