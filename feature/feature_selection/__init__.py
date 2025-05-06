"""
Feature Selection module for Numerai Crypto

This module provides tools for selecting the most relevant features using various methods,
including correlation analysis, model-based importance, and permutation importance.
"""

from .feature_selector import FeatureSelector

__all__ = ['FeatureSelector']