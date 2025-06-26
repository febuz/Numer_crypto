#!/usr/bin/env python3
"""
GPU-Accelerated Mathematical Transformations

This module provides GPU-accelerated mathematical transformations for feature engineering,
including basic transforms, trigonometric transforms, and polynomial/interaction features.
"""

import os
import sys
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Import local modules
from utils.gpu.memory_utils import (
    cuda_available, cupy_available, torch_available,
    detect_gpu_libraries, clear_gpu_memory, get_gpu_memory_usage,
    check_gpu_memory_availability
)
from utils.gpu.data_conversion import to_gpu, to_cpu, chunk_array_for_gpu

# Run detection at module import if not already done
if not cuda_available:
    detect_gpu_libraries()

def basic_transforms(data: np.ndarray, feature_names: List[str], 
                    max_chunk_size: int = 1000000) -> Tuple[np.ndarray, List[str]]:
    """
    Apply basic mathematical transformations to features
    
    Transformations include:
    - Log transforms (log, log1p)
    - Exponential transforms (exp, expm1)
    - Square, square root, cube, cube root
    - Reciprocal, sigmoid
    
    Args:
        data: Input array of shape (n_samples, n_features)
        feature_names: Names of the input features
        max_chunk_size: Maximum chunk size for GPU processing
        
    Returns:
        Tuple of (transformed_data, transformed_feature_names)
    """
    # Check if GPU is available and has enough memory
    use_gpu = cuda_available and check_gpu_memory_availability(data.shape, "basic")
    
    if use_gpu:
        logger.info(f"Applying GPU-accelerated basic transforms to {data.shape[1]} features")
        
        # Check if we need to chunk the data
        if data.shape[0] > max_chunk_size:
            # Process in chunks
            chunks = chunk_array_for_gpu(data, max_chunk_size)
            results = []
            
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)} with shape {chunk.shape}")
                # Process each chunk
                chunk_result, _ = _process_basic_transforms_gpu(chunk, feature_names)
                results.append(chunk_result)
                
                # Clear GPU memory after each chunk
                clear_gpu_memory()
            
            # Concatenate results
            transformed_data = np.vstack(results)
            
            # Get feature names from the first chunk processing
            _, transformed_feature_names = _process_basic_transforms_gpu(chunks[0][:1], feature_names)
            
            return transformed_data, transformed_feature_names
        else:
            # Process without chunking
            return _process_basic_transforms_gpu(data, feature_names)
    else:
        logger.info(f"Applying CPU basic transforms to {data.shape[1]} features")
        return _process_basic_transforms_cpu(data, feature_names)

def _process_basic_transforms_gpu(data: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Process basic transforms using GPU acceleration
    
    Args:
        data: Input data array
        feature_names: Names of the input features
        
    Returns:
        Tuple of (transformed_data, transformed_feature_names)
    """
    start_time = time.time()
    
    # Convert to GPU
    gpu_data = to_gpu(data)
    
    # List to store transformed features and their names
    transformed_features = [gpu_data]  # Start with original features
    transformed_names = feature_names.copy()
    
    # Apply transforms
    if cupy_available:
        import cupy as cp
        
        # Log transforms
        log_data = cp.log(cp.clip(gpu_data, 1e-10, None))
        log1p_data = cp.log1p(cp.abs(gpu_data))
        
        # Exponential transforms
        exp_data = cp.exp(cp.clip(gpu_data, None, 10))
        expm1_data = cp.expm1(cp.clip(gpu_data, None, 10))
        
        # Power transforms
        square_data = cp.square(gpu_data)
        sqrt_data = cp.sqrt(cp.abs(gpu_data))
        cube_data = cp.power(gpu_data, 3)
        cbrt_data = cp.cbrt(gpu_data)
        
        # Other transforms
        recip_data = 1.0 / (cp.abs(gpu_data) + 1e-10)
        sigmoid_data = 1.0 / (1.0 + cp.exp(-gpu_data))
        
    elif torch_available:
        import torch
        
        # Log transforms
        log_data = torch.log(torch.clamp(gpu_data, min=1e-10))
        log1p_data = torch.log1p(torch.abs(gpu_data))
        
        # Exponential transforms
        exp_data = torch.exp(torch.clamp(gpu_data, max=10))
        expm1_data = torch.expm1(torch.clamp(gpu_data, max=10))
        
        # Power transforms
        square_data = torch.square(gpu_data)
        sqrt_data = torch.sqrt(torch.abs(gpu_data))
        cube_data = torch.pow(gpu_data, 3)
        cbrt_data = torch.sign(gpu_data) * torch.pow(torch.abs(gpu_data), 1.0/3.0)
        
        # Other transforms
        recip_data = 1.0 / (torch.abs(gpu_data) + 1e-10)
        sigmoid_data = 1.0 / (1.0 + torch.exp(-gpu_data))
    
    # Add transformed features to the list
    transformed_features.extend([
        log_data, log1p_data, 
        exp_data, expm1_data,
        square_data, sqrt_data, 
        cube_data, cbrt_data,
        recip_data, sigmoid_data
    ])
    
    # Add transformed feature names
    for prefix in ["log_", "log1p_", "exp_", "expm1_", "square_", "sqrt_", "cube_", "cbrt_", "recip_", "sigmoid_"]:
        transformed_names.extend([f"{prefix}{name}" for name in feature_names])
    
    # Concatenate all features
    if cupy_available:
        import cupy as cp
        all_features = cp.hstack(transformed_features)
    elif torch_available:
        import torch
        all_features = torch.cat(transformed_features, dim=1)
    
    # Convert back to CPU
    result = to_cpu(all_features)
    
    # Log performance
    elapsed = time.time() - start_time
    logger.info(f"GPU basic transforms: {data.shape[1]} features -> {result.shape[1]} features in {elapsed:.2f}s")
    
    return result, transformed_names

def _process_basic_transforms_cpu(data: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Process basic transforms using CPU (fallback)
    
    Args:
        data: Input data array
        feature_names: Names of the input features
        
    Returns:
        Tuple of (transformed_data, transformed_feature_names)
    """
    start_time = time.time()
    
    # List to store transformed features and their names
    transformed_features = [data]  # Start with original features
    transformed_names = feature_names.copy()
    
    # Apply transforms
    # Log transforms
    log_data = np.log(np.clip(data, 1e-10, None))
    log1p_data = np.log1p(np.abs(data))
    
    # Exponential transforms
    exp_data = np.exp(np.clip(data, None, 10))
    expm1_data = np.expm1(np.clip(data, None, 10))
    
    # Power transforms
    square_data = np.square(data)
    sqrt_data = np.sqrt(np.abs(data))
    cube_data = np.power(data, 3)
    cbrt_data = np.cbrt(data)
    
    # Other transforms
    recip_data = 1.0 / (np.abs(data) + 1e-10)
    sigmoid_data = 1.0 / (1.0 + np.exp(-data))
    
    # Add transformed features to the list
    transformed_features.extend([
        log_data, log1p_data, 
        exp_data, expm1_data,
        square_data, sqrt_data, 
        cube_data, cbrt_data,
        recip_data, sigmoid_data
    ])
    
    # Add transformed feature names
    for prefix in ["log_", "log1p_", "exp_", "expm1_", "square_", "sqrt_", "cube_", "cbrt_", "recip_", "sigmoid_"]:
        transformed_names.extend([f"{prefix}{name}" for name in feature_names])
    
    # Concatenate all features
    all_features = np.hstack(transformed_features)
    
    # Log performance
    elapsed = time.time() - start_time
    logger.info(f"CPU basic transforms: {data.shape[1]} features -> {all_features.shape[1]} features in {elapsed:.2f}s")
    
    return all_features, transformed_names

def trigonometric_transforms(data: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Apply trigonometric transformations to features
    
    Transformations include:
    - Sine, cosine, tangent
    - Hyperbolic sine, cosine, tangent
    
    Args:
        data: Input array of shape (n_samples, n_features)
        feature_names: Names of the input features
        
    Returns:
        Tuple of (transformed_data, transformed_feature_names)
    """
    # Check if GPU is available and has enough memory
    use_gpu = cuda_available and check_gpu_memory_availability(data.shape, "trigonometric")
    
    if use_gpu:
        logger.info(f"Applying GPU-accelerated trigonometric transforms to {data.shape[1]} features")
        return _process_trigonometric_transforms_gpu(data, feature_names)
    else:
        logger.info(f"Applying CPU trigonometric transforms to {data.shape[1]} features")
        return _process_trigonometric_transforms_cpu(data, feature_names)

def _process_trigonometric_transforms_gpu(data: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Process trigonometric transforms using GPU acceleration
    
    Args:
        data: Input data array
        feature_names: Names of the input features
        
    Returns:
        Tuple of (transformed_data, transformed_feature_names)
    """
    start_time = time.time()
    
    # Convert to GPU
    gpu_data = to_gpu(data)
    
    # Scale data to avoid overflow (clip to reasonable range)
    if cupy_available:
        import cupy as cp
        scaled_data = cp.clip(gpu_data, -10, 10)
    elif torch_available:
        import torch
        scaled_data = torch.clamp(gpu_data, min=-10, max=10)
    
    # List to store transformed features and their names
    transformed_features = [gpu_data]  # Start with original features
    transformed_names = feature_names.copy()
    
    # Apply transforms
    if cupy_available:
        import cupy as cp
        
        # Trigonometric transforms
        sin_data = cp.sin(scaled_data)
        cos_data = cp.cos(scaled_data)
        tan_data = cp.tan(cp.clip(scaled_data, -1.5, 1.5))  # Clip to avoid tan(π/2)
        
        # Hyperbolic transforms
        sinh_data = cp.sinh(cp.clip(scaled_data, -5, 5))  # Clip to avoid overflow
        cosh_data = cp.cosh(cp.clip(scaled_data, -5, 5))
        tanh_data = cp.tanh(scaled_data)
        
    elif torch_available:
        import torch
        
        # Trigonometric transforms
        sin_data = torch.sin(scaled_data)
        cos_data = torch.cos(scaled_data)
        tan_data = torch.tan(torch.clamp(scaled_data, min=-1.5, max=1.5))
        
        # Hyperbolic transforms
        sinh_data = torch.sinh(torch.clamp(scaled_data, min=-5, max=5))
        cosh_data = torch.cosh(torch.clamp(scaled_data, min=-5, max=5))
        tanh_data = torch.tanh(scaled_data)
    
    # Add transformed features to the list
    transformed_features.extend([
        sin_data, cos_data, tan_data,
        sinh_data, cosh_data, tanh_data
    ])
    
    # Add transformed feature names
    for prefix in ["sin_", "cos_", "tan_", "sinh_", "cosh_", "tanh_"]:
        transformed_names.extend([f"{prefix}{name}" for name in feature_names])
    
    # Concatenate all features
    if cupy_available:
        import cupy as cp
        all_features = cp.hstack(transformed_features)
    elif torch_available:
        import torch
        all_features = torch.cat(transformed_features, dim=1)
    
    # Convert back to CPU
    result = to_cpu(all_features)
    
    # Log performance
    elapsed = time.time() - start_time
    logger.info(f"GPU trigonometric transforms: {data.shape[1]} features -> {result.shape[1]} features in {elapsed:.2f}s")
    
    return result, transformed_names

def _process_trigonometric_transforms_cpu(data: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Process trigonometric transforms using CPU (fallback)
    
    Args:
        data: Input data array
        feature_names: Names of the input features
        
    Returns:
        Tuple of (transformed_data, transformed_feature_names)
    """
    start_time = time.time()
    
    # Scale data to avoid overflow
    scaled_data = np.clip(data, -10, 10)
    
    # List to store transformed features and their names
    transformed_features = [data]  # Start with original features
    transformed_names = feature_names.copy()
    
    # Apply transforms
    # Trigonometric transforms
    sin_data = np.sin(scaled_data)
    cos_data = np.cos(scaled_data)
    tan_data = np.tan(np.clip(scaled_data, -1.5, 1.5))
    
    # Hyperbolic transforms
    sinh_data = np.sinh(np.clip(scaled_data, -5, 5))
    cosh_data = np.cosh(np.clip(scaled_data, -5, 5))
    tanh_data = np.tanh(scaled_data)
    
    # Add transformed features to the list
    transformed_features.extend([
        sin_data, cos_data, tan_data,
        sinh_data, cosh_data, tanh_data
    ])
    
    # Add transformed feature names
    for prefix in ["sin_", "cos_", "tan_", "sinh_", "cosh_", "tanh_"]:
        transformed_names.extend([f"{prefix}{name}" for name in feature_names])
    
    # Concatenate all features
    all_features = np.hstack(transformed_features)
    
    # Log performance
    elapsed = time.time() - start_time
    logger.info(f"CPU trigonometric transforms: {data.shape[1]} features -> {all_features.shape[1]} features in {elapsed:.2f}s")
    
    return all_features, transformed_names

def polynomial_transforms(data: np.ndarray, feature_names: List[str], 
                         max_degree: int = 3) -> Tuple[np.ndarray, List[str]]:
    """
    Apply polynomial transformations to features
    
    Args:
        data: Input array of shape (n_samples, n_features)
        feature_names: Names of the input features
        max_degree: Maximum polynomial degree
        
    Returns:
        Tuple of (transformed_data, transformed_feature_names)
    """
    # Check if data is too large for polynomial expansion
    if data.shape[1] > 50 and max_degree > 2:
        logger.warning(f"Data has {data.shape[1]} features. Limiting to degree 2 to avoid memory overflow.")
        max_degree = 2
    
    # Check if GPU is available and has enough memory
    use_gpu = cuda_available and check_gpu_memory_availability(data.shape, "polynomial")
    
    if use_gpu:
        logger.info(f"Applying GPU-accelerated polynomial transforms (degree {max_degree}) to {data.shape[1]} features")
        
        # Try using scikit-learn's PolynomialFeatures on GPU
        try:
            from sklearn.preprocessing import PolynomialFeatures
            
            # Convert data to CPU for scikit-learn
            poly = PolynomialFeatures(degree=max_degree, include_bias=False)
            
            # Fit and transform on CPU (scikit-learn doesn't support GPU)
            start_time = time.time()
            poly_features = poly.fit_transform(data)
            
            # Get feature names
            if hasattr(poly, 'get_feature_names_out'):
                poly_feature_names = poly.get_feature_names_out(feature_names)
            else:
                poly_feature_names = poly.get_feature_names(feature_names)
            
            elapsed = time.time() - start_time
            logger.info(f"Polynomial transforms: {data.shape[1]} features -> {poly_features.shape[1]} features in {elapsed:.2f}s")
            
            return poly_features, list(poly_feature_names)
            
        except Exception as e:
            logger.warning(f"Error using scikit-learn for polynomial features: {e}")
            logger.info("Falling back to custom implementation")
            
            # Fall back to our custom implementation
            return _custom_polynomial_transforms_gpu(data, feature_names, max_degree)
    else:
        logger.info(f"Applying CPU polynomial transforms (degree {max_degree}) to {data.shape[1]} features")
        
        # Use scikit-learn
        try:
            from sklearn.preprocessing import PolynomialFeatures
            
            poly = PolynomialFeatures(degree=max_degree, include_bias=False)
            
            start_time = time.time()
            poly_features = poly.fit_transform(data)
            
            # Get feature names
            if hasattr(poly, 'get_feature_names_out'):
                poly_feature_names = poly.get_feature_names_out(feature_names)
            else:
                poly_feature_names = poly.get_feature_names(feature_names)
            
            elapsed = time.time() - start_time
            logger.info(f"CPU polynomial transforms: {data.shape[1]} features -> {poly_features.shape[1]} features in {elapsed:.2f}s")
            
            return poly_features, list(poly_feature_names)
            
        except Exception as e:
            logger.warning(f"Error using scikit-learn for polynomial features: {e}")
            logger.info("Falling back to custom implementation")
            
            # Fall back to custom implementation
            return _custom_polynomial_transforms_cpu(data, feature_names, max_degree)

def _custom_polynomial_transforms_gpu(data: np.ndarray, feature_names: List[str], 
                                    max_degree: int) -> Tuple[np.ndarray, List[str]]:
    """
    Custom GPU implementation of polynomial transformations
    
    Args:
        data: Input data array
        feature_names: Names of the input features
        max_degree: Maximum polynomial degree
        
    Returns:
        Tuple of (transformed_data, transformed_feature_names)
    """
    start_time = time.time()
    
    # Convert to GPU
    gpu_data = to_gpu(data)
    
    # List to store transformed features and their names
    transformed_features = [gpu_data]  # Start with original features
    transformed_names = feature_names.copy()
    
    # Get framework
    if cupy_available:
        import cupy as cp
        xp = cp
    elif torch_available:
        import torch
        xp = torch
    
    # For degree 2 and above, add squared terms
    if max_degree >= 2:
        squared_features = xp.square(gpu_data)
        transformed_features.append(squared_features)
        transformed_names.extend([f"{name}^2" for name in feature_names])
        
        # Add pairwise interaction terms
        n_features = gpu_data.shape[1]
        for i in range(n_features):
            for j in range(i+1, n_features):
                interaction = gpu_data[:, i:i+1] * gpu_data[:, j:j+1]
                transformed_features.append(interaction)
                transformed_names.append(f"{feature_names[i]}*{feature_names[j]}")
    
    # For degree 3, add cubic terms
    if max_degree >= 3:
        cubed_features = xp.power(gpu_data, 3)
        transformed_features.append(cubed_features)
        transformed_names.extend([f"{name}^3" for name in feature_names])
        
        # Add feature^2 * feature interactions (very costly, limit to small datasets)
        if n_features <= 20:
            for i in range(n_features):
                squared_i = gpu_data[:, i:i+1] ** 2
                for j in range(n_features):
                    if i != j:
                        interaction = squared_i * gpu_data[:, j:j+1]
                        transformed_features.append(interaction)
                        transformed_names.append(f"{feature_names[i]}^2*{feature_names[j]}")
    
    # Concatenate all features
    if cupy_available:
        all_features = cp.hstack(transformed_features)
    elif torch_available:
        all_features = torch.cat(transformed_features, dim=1)
    
    # Convert back to CPU
    result = to_cpu(all_features)
    
    # Log performance
    elapsed = time.time() - start_time
    logger.info(f"GPU custom polynomial transforms: {data.shape[1]} features -> {result.shape[1]} features in {elapsed:.2f}s")
    
    return result, transformed_names

def _custom_polynomial_transforms_cpu(data: np.ndarray, feature_names: List[str], 
                                    max_degree: int) -> Tuple[np.ndarray, List[str]]:
    """
    Custom CPU implementation of polynomial transformations
    
    Args:
        data: Input data array
        feature_names: Names of the input features
        max_degree: Maximum polynomial degree
        
    Returns:
        Tuple of (transformed_data, transformed_feature_names)
    """
    start_time = time.time()
    
    # List to store transformed features and their names
    transformed_features = [data]  # Start with original features
    transformed_names = feature_names.copy()
    
    n_features = data.shape[1]
    
    # For degree 2 and above, add squared terms
    if max_degree >= 2:
        squared_features = np.square(data)
        transformed_features.append(squared_features)
        transformed_names.extend([f"{name}^2" for name in feature_names])
        
        # Add pairwise interaction terms
        for i in range(n_features):
            for j in range(i+1, n_features):
                interaction = data[:, i:i+1] * data[:, j:j+1]
                transformed_features.append(interaction)
                transformed_names.append(f"{feature_names[i]}*{feature_names[j]}")
    
    # For degree 3, add cubic terms
    if max_degree >= 3:
        cubed_features = np.power(data, 3)
        transformed_features.append(cubed_features)
        transformed_names.extend([f"{name}^3" for name in feature_names])
        
        # Add feature^2 * feature interactions (very costly, limit to small datasets)
        if n_features <= 20:
            for i in range(n_features):
                squared_i = data[:, i:i+1] ** 2
                for j in range(n_features):
                    if i != j:
                        interaction = squared_i * data[:, j:j+1]
                        transformed_features.append(interaction)
                        transformed_names.append(f"{feature_names[i]}^2*{feature_names[j]}")
    
    # Concatenate all features
    all_features = np.hstack(transformed_features)
    
    # Log performance
    elapsed = time.time() - start_time
    logger.info(f"CPU custom polynomial transforms: {data.shape[1]} features -> {all_features.shape[1]} features in {elapsed:.2f}s")
    
    return all_features, transformed_names

def interaction_transforms(data: np.ndarray, feature_names: List[str], 
                          max_interactions: int = 2000) -> Tuple[np.ndarray, List[str]]:
    """
    Generate interaction features (pairwise products)
    
    Args:
        data: Input array of shape (n_samples, n_features)
        feature_names: Names of the input features
        max_interactions: Maximum number of interaction features to create
        
    Returns:
        Tuple of (transformed_data, transformed_feature_names)
    """
    # Check if GPU is available and has enough memory
    use_gpu = cuda_available and check_gpu_memory_availability(data.shape, "interaction")
    
    if use_gpu:
        logger.info(f"Applying GPU-accelerated interaction transforms to {data.shape[1]} features")
        return _process_interaction_transforms_gpu(data, feature_names, max_interactions)
    else:
        logger.info(f"Applying CPU interaction transforms to {data.shape[1]} features")
        return _process_interaction_transforms_cpu(data, feature_names, max_interactions)

def _process_interaction_transforms_gpu(data: np.ndarray, feature_names: List[str], 
                                       max_interactions: int) -> Tuple[np.ndarray, List[str]]:
    """
    Process interaction transforms using GPU acceleration
    
    Args:
        data: Input data array
        feature_names: Names of the input features
        max_interactions: Maximum number of interaction features
        
    Returns:
        Tuple of (transformed_data, transformed_feature_names)
    """
    start_time = time.time()
    
    # Convert to GPU
    gpu_data = to_gpu(data)
    
    # List to store transformed features and their names
    transformed_features = [gpu_data]  # Start with original features
    transformed_names = feature_names.copy()
    
    # Determine pairs for interaction features
    n_features = data.shape[1]
    
    # Generate all possible pairs
    all_pairs = []
    for i in range(n_features):
        for j in range(i+1, n_features):
            all_pairs.append((i, j))
    
    # If too many pairs, select a subset
    if len(all_pairs) > max_interactions:
        import random
        random.seed(42)  # For reproducibility
        selected_pairs = random.sample(all_pairs, max_interactions)
    else:
        selected_pairs = all_pairs
    
    logger.info(f"Generating {len(selected_pairs)} interaction features")
    
    # Generate interaction features
    if cupy_available:
        import cupy as cp
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        for batch_start in range(0, len(selected_pairs), batch_size):
            batch_end = min(batch_start + batch_size, len(selected_pairs))
            batch_pairs = selected_pairs[batch_start:batch_end]
            
            # Create interactions for this batch
            batch_interactions = []
            batch_names = []
            
            for i, j in batch_pairs:
                interaction = gpu_data[:, i:i+1] * gpu_data[:, j:j+1]
                batch_interactions.append(interaction)
                batch_names.append(f"{feature_names[i]}*{feature_names[j]}")
            
            # Combine batch interactions
            batch_features = cp.hstack(batch_interactions)
            
            # Add to transformed features
            transformed_features.append(batch_features)
            transformed_names.extend(batch_names)
            
            # Clear memory
            clear_gpu_memory()
        
        # Concatenate all features
        all_features = cp.hstack(transformed_features)
        
    elif torch_available:
        import torch
        
        # Process in batches
        batch_size = 1000
        for batch_start in range(0, len(selected_pairs), batch_size):
            batch_end = min(batch_start + batch_size, len(selected_pairs))
            batch_pairs = selected_pairs[batch_start:batch_end]
            
            # Create interactions for this batch
            batch_interactions = []
            batch_names = []
            
            for i, j in batch_pairs:
                interaction = gpu_data[:, i:i+1] * gpu_data[:, j:j+1]
                batch_interactions.append(interaction)
                batch_names.append(f"{feature_names[i]}*{feature_names[j]}")
            
            # Combine batch interactions
            batch_features = torch.cat(batch_interactions, dim=1)
            
            # Add to transformed features
            transformed_features.append(batch_features)
            transformed_names.extend(batch_names)
            
            # Clear memory
            clear_gpu_memory()
        
        # Concatenate all features
        all_features = torch.cat(transformed_features, dim=1)
    
    # Convert back to CPU
    result = to_cpu(all_features)
    
    # Log performance
    elapsed = time.time() - start_time
    logger.info(f"GPU interaction transforms: {data.shape[1]} features -> {result.shape[1]} features in {elapsed:.2f}s")
    
    return result, transformed_names

def _process_interaction_transforms_cpu(data: np.ndarray, feature_names: List[str], 
                                       max_interactions: int) -> Tuple[np.ndarray, List[str]]:
    """
    Process interaction transforms using CPU (fallback)
    
    Args:
        data: Input data array
        feature_names: Names of the input features
        max_interactions: Maximum number of interaction features
        
    Returns:
        Tuple of (transformed_data, transformed_feature_names)
    """
    start_time = time.time()
    
    # List to store transformed features and their names
    transformed_features = [data]  # Start with original features
    transformed_names = feature_names.copy()
    
    # Determine pairs for interaction features
    n_features = data.shape[1]
    
    # Generate all possible pairs
    all_pairs = []
    for i in range(n_features):
        for j in range(i+1, n_features):
            all_pairs.append((i, j))
    
    # If too many pairs, select a subset
    if len(all_pairs) > max_interactions:
        import random
        random.seed(42)  # For reproducibility
        selected_pairs = random.sample(all_pairs, max_interactions)
    else:
        selected_pairs = all_pairs
    
    logger.info(f"Generating {len(selected_pairs)} interaction features")
    
    # Process in batches to avoid memory issues
    batch_size = 1000
    for batch_start in range(0, len(selected_pairs), batch_size):
        batch_end = min(batch_start + batch_size, len(selected_pairs))
        batch_pairs = selected_pairs[batch_start:batch_end]
        
        # Create interactions for this batch
        batch_interactions = []
        batch_names = []
        
        for i, j in batch_pairs:
            interaction = data[:, i:i+1] * data[:, j:j+1]
            batch_interactions.append(interaction)
            batch_names.append(f"{feature_names[i]}*{feature_names[j]}")
        
        # Combine batch interactions
        batch_features = np.hstack(batch_interactions)
        
        # Add to transformed features
        transformed_features.append(batch_features)
        transformed_names.extend(batch_names)
    
    # Concatenate all features
    all_features = np.hstack(transformed_features)
    
    # Log performance
    elapsed = time.time() - start_time
    logger.info(f"CPU interaction transforms: {data.shape[1]} features -> {all_features.shape[1]} features in {elapsed:.2f}s")
    
    return all_features, transformed_names

if __name__ == "__main__":
    # Test the transformations
    
    # Create test data
    np.random.seed(42)
    X = np.random.random((1000, 10)).astype(np.float32)
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Test basic transforms
    X_basic, basic_names = basic_transforms(X, feature_names)
    print(f"Basic transforms: {X.shape} -> {X_basic.shape}")
    print(f"First 5 transformed feature names: {basic_names[:5]}")
    
    # Test trigonometric transforms
    X_trig, trig_names = trigonometric_transforms(X, feature_names)
    print(f"Trigonometric transforms: {X.shape} -> {X_trig.shape}")
    print(f"First 5 transformed feature names: {trig_names[:5]}")
    
    # Test polynomial transforms
    X_poly, poly_names = polynomial_transforms(X, feature_names, max_degree=2)
    print(f"Polynomial transforms: {X.shape} -> {X_poly.shape}")
    print(f"First 5 transformed feature names: {poly_names[:5]}")
    
    # Test interaction transforms
    X_inter, inter_names = interaction_transforms(X, feature_names, max_interactions=10)
    print(f"Interaction transforms: {X.shape} -> {X_inter.shape}")
    print(f"First 5 transformed feature names: {inter_names[:5]}")