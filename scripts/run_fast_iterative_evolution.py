#!/usr/bin/env python3
"""
Fast Iterative Feature Evolution Pipeline:
1. Generate simple but effective features from current dataset
2. Apply conservative reduction (max 50% reduction)
3. Use the reduced set as base for next iteration
4. Repeat for multiple cycles to evolve better features

This creates a FAST feature evolution system focused on speed and effectiveness.
"""

import os
import sys
import argparse
import logging
import time
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from scripts.run_conservative_reducer import ConservativeReducer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class FastIterativeEvolution:
    """
    Fast iterative feature evolution with simple but effective feature generation
    """
    
    def __init__(self,
                 checkpoint_dir: str,
                 max_iterations: int = 5,
                 features_per_iteration: int = 5000,
                 max_reduction: float = 0.45,
                 min_features_per_iteration: int = 1000,
                 memory_limit_gb: float = 120.0,
                 sample_size: int = 0,  # 0 means use full dataset
                 use_gpu: bool = True,
                 use_random_baselines: bool = True):
        """Initialize fast iterative evolution"""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_iterations = max_iterations
        self.features_per_iteration = features_per_iteration
        self.max_reduction = max_reduction
        self.min_features_per_iteration = min_features_per_iteration
        self.memory_limit_gb = memory_limit_gb
        self.sample_size = sample_size
        self.use_gpu = use_gpu
        self.use_random_baselines = use_random_baselines
        
        # Check GPU availability
        self.gpu_available = self._check_gpu_availability()
        
        # Initialize conservative reducer (it has built-in GPU support)
        self.conservative_reducer = ConservativeReducer(
            checkpoint_dir=str(self.checkpoint_dir / "reducer_checkpoints"),
            max_reduction=max_reduction,
            min_features=min_features_per_iteration,
            memory_limit_gb=memory_limit_gb,
            sample_size=sample_size  # 0 means use full dataset
        )
        
        self.evolution_history = []
        
        logger.info("🚀 FAST ITERATIVE FEATURE EVOLUTION INITIALIZED")
        logger.info(f"  🔄 Max iterations: {max_iterations}")
        logger.info(f"  🎯 Features per iteration: {features_per_iteration}")
        logger.info(f"  🛡️  Max reduction: {max_reduction*100:.0f}%")
        logger.info(f"  🔢 Min features per iteration: {min_features_per_iteration}")
        logger.info(f"  💾 Memory limit: {memory_limit_gb}GB")
        
        # Clear indication of whether using sample or full dataset
        if sample_size == 0:
            logger.info(f"  📊 Using FULL dataset (no sampling)")
        else:
            logger.info(f"  📊 Sample size: {sample_size:,} rows")

    def get_memory_usage_gb(self) -> float:
        """Get current memory usage in GB"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            process_memory_gb = memory_info.rss / 1024 / 1024 / 1024
            
            # Also get system memory info
            system_memory = psutil.virtual_memory()
            system_used_gb = (system_memory.total - system_memory.available) / 1024 / 1024 / 1024
            system_total_gb = system_memory.total / 1024 / 1024 / 1024
            
            # Log detailed memory info periodically
            if not hasattr(self, '_memory_log_counter'):
                self._memory_log_counter = 0
            
            self._memory_log_counter += 1
            if self._memory_log_counter % 10 == 0:
                logger.info(f"Memory usage - Process: {process_memory_gb:.1f}GB, System: {system_used_gb:.1f}GB/{system_total_gb:.1f}GB ({system_memory.percent:.1f}%)")
            
            return process_memory_gb
        except ImportError:
            return 0.0
    
    def manage_memory(self, force_cleanup: bool = False, threshold_gb: float = 500.0) -> bool:
        """
        Actively manage memory to prevent OOM errors
        
        Args:
            force_cleanup: Force memory cleanup regardless of current usage
            threshold_gb: Memory threshold in GB to trigger cleanup
            
        Returns:
            True if cleanup was performed, False otherwise
        """
        try:
            import psutil
            
            # Get current memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            process_memory_gb = memory_info.rss / 1024 / 1024 / 1024
            
            # Get system memory status
            system_memory = psutil.virtual_memory()
            system_available_gb = system_memory.available / 1024 / 1024 / 1024
            system_used_percent = system_memory.percent
            
            # Check if cleanup is needed
            if force_cleanup:
                cleanup_needed = True
                reason = "forced cleanup"
            elif process_memory_gb > threshold_gb:
                cleanup_needed = True
                reason = f"process memory ({process_memory_gb:.1f}GB) above threshold ({threshold_gb:.1f}GB)"
            elif system_available_gb < 60:  # Critical - less than 60GB free
                cleanup_needed = True
                reason = f"critically low system memory ({system_available_gb:.1f}GB free)"
            elif system_used_percent > 90:  # Critical - more than 90% used
                cleanup_needed = True
                reason = f"critically high system memory usage ({system_used_percent:.1f}%)"
            else:
                cleanup_needed = False
                reason = "not needed"
            
            # Perform cleanup if needed
            if cleanup_needed:
                logger.info(f"Memory cleanup triggered: {reason}")
                
                # Aggressive garbage collection
                import gc
                gc.collect()
                
                # Clear PyTorch cache if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("Cleared PyTorch CUDA cache")
                except (ImportError, Exception):
                    pass
                
                # Clear CuPy cache if available
                try:
                    import cupy as cp
                    mempool = cp.get_default_memory_pool()
                    pinned_mempool = cp.get_default_pinned_memory_pool()
                    mempool.free_all_blocks()
                    pinned_mempool.free_all_blocks()
                    logger.info("Cleared CuPy memory pools")
                except (ImportError, Exception):
                    pass
                
                # Get memory usage after cleanup
                memory_info = process.memory_info()
                process_memory_gb_after = memory_info.rss / 1024 / 1024 / 1024
                
                system_memory = psutil.virtual_memory()
                system_available_gb_after = system_memory.available / 1024 / 1024 / 1024
                
                # Log memory status after cleanup
                logger.info(f"Memory after cleanup - Process: {process_memory_gb_after:.1f}GB (freed {process_memory_gb - process_memory_gb_after:.1f}GB), System free: {system_available_gb_after:.1f}GB")
                
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Memory management error: {e}")
            return False

    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available"""
        if not self.use_gpu:
            return False
        
        try:
            import cupy as cp
            
            # Set proper memory pool limits for 24GB GPUs
            mempool = cp.get_default_memory_pool()
            mem_info = cp.cuda.runtime.memGetInfo()
            free_bytes = mem_info[0]
            # Use 80% of available memory (up to 20GB for 24GB GPUs)
            max_memory = min(free_bytes * 0.8, 20 * 1024**3)
            mempool.set_limit(size=int(max_memory))
            
            # Test GPU availability
            cp.array([1, 2, 3])
            logger.info(f"🚀 GPU acceleration available (CuPy) - Memory limit: {max_memory/(1024**3):.1f}GB")
            return True
        except ImportError:
            logger.info("⚠️  CuPy not available - using CPU")
            return False
        except Exception as e:
            logger.warning(f"⚠️  GPU test failed: {e} - using CPU")
            return False

    def _gpu_feature_interactions(self, df_np: np.ndarray, col_names: List[str], target_features: int) -> Tuple[np.ndarray, List[str]]:
        """Ultra-fast GPU feature interactions using new accelerator"""
        try:
            from features.gpu_math_accelerator import GPUMathAccelerator
            logger.info("🚀 Using ultra-fast GPU feature interactions...")
            
            # Initialize GPU math accelerator with proper memory pool
            math_accelerator = GPUMathAccelerator(memory_pool_size=int(20 * 1024**3))
            
            # Generate interaction features only
            result_data, result_names = math_accelerator.gpu_interaction_transforms(
                data=df_np,
                feature_names=col_names,
                max_interactions=target_features
            )
            
            logger.info(f"✅ Ultra-fast GPU generated {len(result_names)} interaction features")
            return result_data, result_names
            
        except ImportError:
            logger.warning("GPU Math Accelerator not available, falling back to basic GPU interactions")
            return self._basic_gpu_feature_interactions(df_np, col_names, target_features)
        except Exception as e:
            logger.error(f"GPU Math Accelerator failed: {e}, falling back to basic GPU interactions")
            return self._basic_gpu_feature_interactions(df_np, col_names, target_features)
    
    def _basic_gpu_feature_interactions(self, df_np: np.ndarray, col_names: List[str], target_features: int) -> Tuple[np.ndarray, List[str]]:
        """Basic GPU feature interactions (fallback)"""
        import cupy as cp
        
        logger.info("🚀 Using basic GPU for feature interactions...")
        
        # Transfer to GPU
        gpu_data = cp.array(df_np.astype(np.float32))
        n_samples, n_features = gpu_data.shape
        
        new_features = []
        new_names = []
        
        # Calculate how many interactions we can do
        max_pairs = min(target_features // 3, 1000)  # 3 operations per pair
        
        # Select feature pairs efficiently
        selected_indices = []
        for i in range(min(50, n_features)):
            for j in range(i + 1, min(i + 20, n_features)):
                if len(selected_indices) >= max_pairs:
                    break
                selected_indices.append((i, j))
            if len(selected_indices) >= max_pairs:
                break
        
        # Batch GPU operations
        batch_size = 100
        for batch_start in range(0, len(selected_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(selected_indices))
            batch_indices = selected_indices[batch_start:batch_end]
            
            # Process batch on GPU
            for i, j in batch_indices:
                col1_name = col_names[i]
                col2_name = col_names[j]
                
                col1_gpu = gpu_data[:, i]
                col2_gpu = gpu_data[:, j]
                
                # Addition
                add_result = col1_gpu + col2_gpu
                new_features.append(cp.asnumpy(add_result))
                new_names.append(f"gpu_add_{col1_name}_{col2_name}")
                
                # Multiplication
                mul_result = col1_gpu * col2_gpu
                new_features.append(cp.asnumpy(mul_result))
                new_names.append(f"gpu_mul_{col1_name}_{col2_name}")
                
                # Safe division
                div_result = col1_gpu / (col2_gpu + 1e-8)
                new_features.append(cp.asnumpy(div_result))
                new_names.append(f"gpu_div_{col1_name}_{col2_name}")
                
                if len(new_features) >= target_features:
                    break
            
            if len(new_features) >= target_features:
                break
        
        # Convert to numpy array
        if new_features:
            new_features_array = np.column_stack(new_features[:target_features])
            new_names = new_names[:target_features]
            logger.info(f"✅ Basic GPU generated {len(new_names)} interaction features")
            return new_features_array, new_names
        else:
            return np.array([]).reshape(n_samples, 0), []

    def _gpu_math_transforms(self, df_np: np.ndarray, col_names: List[str], target_features: int) -> Tuple[np.ndarray, List[str]]:
        """Ultra-fast GPU mathematical transformations using new accelerator"""
        try:
            from features.gpu_math_accelerator import GPUMathAccelerator
            logger.info("🚀 Using ultra-fast GPU mathematical transformations...")
            
            # Initialize GPU math accelerator with proper memory pool
            math_accelerator = GPUMathAccelerator(memory_pool_size=int(20 * 1024**3))
            
            # Generate all mathematical transformations in one optimized call
            result_data, result_names = math_accelerator.generate_all_math_transforms(
                data=df_np,
                feature_names=col_names,
                include_basic=True,
                include_trig=True,
                include_poly=True,
                include_interactions=False,  # Handle interactions separately
                max_poly_degree=2,
                max_interactions=0
            )
            
            # Limit to target features if needed
            if result_data.shape[1] > target_features:
                result_data = result_data[:, :target_features]
                result_names = result_names[:target_features]
            
            logger.info(f"✅ Ultra-fast GPU generated {len(result_names)} transform features")
            return result_data, result_names
            
        except ImportError:
            logger.warning("GPU Math Accelerator not available, falling back to basic GPU transforms")
            return self._basic_gpu_math_transforms(df_np, col_names, target_features)
        except Exception as e:
            logger.error(f"GPU Math Accelerator failed: {e}, falling back to basic GPU transforms")
            return self._basic_gpu_math_transforms(df_np, col_names, target_features)
    
    def _basic_gpu_math_transforms(self, df_np: np.ndarray, col_names: List[str], target_features: int) -> Tuple[np.ndarray, List[str]]:
        """Basic GPU mathematical transformations (fallback)"""
        import cupy as cp
        
        logger.info("🚀 Using basic GPU for mathematical transformations...")
        
        # Transfer to GPU
        gpu_data = cp.array(df_np.astype(np.float32))
        n_samples, n_features = gpu_data.shape
        
        new_features = []
        new_names = []
        
        # Select columns for transformations
        transform_cols = min(target_features // 3, n_features)
        
        for i in range(transform_cols):
            if len(new_features) >= target_features:
                break
            
            col_name = col_names[i]
            col_gpu = gpu_data[:, i]
            
            # Square
            sq_result = col_gpu ** 2
            new_features.append(cp.asnumpy(sq_result))
            new_names.append(f"gpu_sq_{col_name}")
            
            if len(new_features) >= target_features:
                break
            
            # Square root of absolute value
            sqrt_result = cp.sqrt(cp.abs(col_gpu))
            new_features.append(cp.asnumpy(sqrt_result))
            new_names.append(f"gpu_sqrt_{col_name}")
            
            if len(new_features) >= target_features:
                break
            
            # Log1p
            log_result = cp.log1p(cp.abs(col_gpu))
            new_features.append(cp.asnumpy(log_result))
            new_names.append(f"gpu_log_{col_name}")
        
        # Convert to numpy array
        if new_features:
            new_features_array = np.column_stack(new_features[:target_features])
            new_names = new_names[:target_features]
            logger.info(f"✅ Basic GPU generated {len(new_names)} transform features")
            return new_features_array, new_names
        else:
            return np.array([]).reshape(n_samples, 0), []

    def generate_fast_features(self, df: pl.DataFrame, iteration: int, max_new_features: Optional[int] = None) -> pl.DataFrame:
        """Generate features using fast mathematical operations"""
        logger.info(f"🚀 ITERATION {iteration}: FAST FEATURE GENERATION")
        logger.info(f"  📊 Input shape: {df.shape}")
        logger.info(f"  💾 Memory: {self.get_memory_usage_gb():.1f}GB")
        
        start_time = time.time()
        
        try:
            # Check system memory before starting
            self.manage_memory(threshold_gb=500)
            
            # Get numeric columns excluding metadata
            excluded_cols = ['era', 'target', 'symbol', 'date', 'asset', '__index_level_0__']
            numeric_cols = [col for col in df.columns 
                          if col not in excluded_cols and df[col].dtype in [
                              pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int16, pl.Int8
                          ]]
            
            logger.info(f"  🔢 Working with {len(numeric_cols)} numeric columns")
            logger.info(f"  🚀 GPU Available: {self.gpu_available}")
            
            # Check for existing random baseline features
            random_features_exist = any(col.startswith("random_baseline_") for col in df.columns)
            
            # Start with original data
            df_expanded = df.clone()
            new_features_count = 0
            
            # Determine target features - use max_new_features if specified (for batch mode)
            if max_new_features is not None:
                target_new_features = min(max_new_features, len(numeric_cols) * 10)
                logger.info(f"  🎯 Batch mode: targeting {target_new_features} new features (max: {max_new_features})")
            else:
                target_new_features = min(self.features_per_iteration, len(numeric_cols) * 20)
            
            # Create random baseline features if this is the first iteration and feature is enabled
            if iteration == 1 and not random_features_exist and self.use_random_baselines:
                logger.info("  🎲 Creating random baseline features for comparison...")
                
                # Generate random features that match statistics of the dataset
                try:
                    # Calculate statistics for the dataset
                    data_sample = df.select(numeric_cols).sample(n=min(50000, df.shape[0]))
                    # Calculate mean and standard deviation
                    means = data_sample.mean()
                    stds = data_sample.std()
                    
                    # Get average statistics
                    avg_mean = means.mean_horizontal().item()
                    avg_std = stds.mean_horizontal().item()
                    
                    logger.info(f"  📊 Dataset statistics: mean={avg_mean:.4f}, std={avg_std:.4f}")
                    
                    # Generate 3 random baseline features
                    import numpy as np
                    np.random.seed(42)  # For reproducibility
                    
                    # Generate random features with matching statistics
                    for i in range(1, 4):
                        random_data = np.random.normal(avg_mean, avg_std, size=df.shape[0])
                        df_expanded = df_expanded.with_columns([
                            pl.Series(f"random_baseline_{i}", random_data)
                        ])
                        new_features_count += 1
                    
                    logger.info(f"  🎲 Created 3 random baseline features with matching statistics")
                    
                    # Clean up memory
                    del data_sample
                    self.manage_memory(threshold_gb=500)
                    
                except Exception as e:
                    logger.warning(f"  ⚠️ Could not create random baseline features: {e}")
            elif iteration == 1 and not random_features_exist and not self.use_random_baselines:
                logger.info("  🎲 Random baseline features are disabled")
            
            # GPU-accelerated feature generation
            if self.gpu_available and len(numeric_cols) >= 2:
                logger.info("  🚀 Using GPU-accelerated feature generation...")
                
                # Convert numeric columns to numpy for GPU processing
                numeric_data = df.select(numeric_cols).to_numpy().astype(np.float32)
                
                # Manage memory after data conversion
                self.manage_memory(threshold_gb=500)
                
                # 1. GPU Cross-feature interactions
                interaction_target = min(target_new_features // 2, 2000)
                gpu_interactions, interaction_names = self._gpu_feature_interactions(
                    numeric_data, numeric_cols, interaction_target
                )
                
                # Manage memory after interactions
                self.manage_memory(threshold_gb=500)
                
                # 2. GPU Mathematical transformations
                remaining_features = target_new_features - len(interaction_names)
                transform_target = min(remaining_features, 1000)
                gpu_transforms, transform_names = self._gpu_math_transforms(
                    numeric_data, numeric_cols, transform_target
                )
                
                # Manage memory after transformations
                self.manage_memory(threshold_gb=500)
                
                # Combine GPU-generated features
                if gpu_interactions.shape[1] > 0:
                    # Add in batches to reduce memory pressure
                    batch_size = 50
                    for batch_start in range(0, len(interaction_names), batch_size):
                        batch_end = min(batch_start + batch_size, len(interaction_names))
                        
                        # Create series objects for this batch
                        series_list = []
                        for i in range(batch_start, batch_end):
                            series_list.append(pl.Series(interaction_names[i], gpu_interactions[:, i]))
                        
                        # Add batch to dataframe
                        df_expanded = df_expanded.with_columns(series_list)
                        new_features_count += len(series_list)
                        
                        # Check memory after each batch
                        if batch_end < len(interaction_names):
                            self.manage_memory(threshold_gb=500)
                
                if gpu_transforms.shape[1] > 0:
                    # Add in batches to reduce memory pressure
                    batch_size = 50
                    for batch_start in range(0, len(transform_names), batch_size):
                        batch_end = min(batch_start + batch_size, len(transform_names))
                        
                        # Create series objects for this batch
                        series_list = []
                        for i in range(batch_start, batch_end):
                            series_list.append(pl.Series(transform_names[i], gpu_transforms[:, i]))
                        
                        # Add batch to dataframe
                        df_expanded = df_expanded.with_columns(series_list)
                        new_features_count += len(series_list)
                        
                        # Check memory after each batch
                        if batch_end < len(transform_names):
                            self.manage_memory(threshold_gb=500)
                
                logger.info(f"  ✅ GPU generated {new_features_count} features")
                
                # Clear GPU memory
                del gpu_interactions
                del gpu_transforms
                self.manage_memory(force_cleanup=True, threshold_gb=500)
            
            # Fallback to CPU if GPU not available or for additional features
            if new_features_count < target_new_features:
                remaining_target = target_new_features - new_features_count
                logger.info(f"  💻 CPU fallback for {remaining_target} additional features...")
                
                # CPU Cross-feature interactions
                if len(numeric_cols) >= 2:
                    interaction_cols = numeric_cols[:min(30, len(numeric_cols))]
                    
                    # Process in batches to manage memory
                    batch_size = 20
                    for batch_start in range(0, min(30, len(interaction_cols)), batch_size):
                        batch_end = min(batch_start + batch_size, len(interaction_cols))
                        batch_cols = interaction_cols[batch_start:batch_end]
                        
                        series_list = []
                        for i, col1 in enumerate(batch_cols):
                            if new_features_count >= target_new_features:
                                break
                            for j, col2 in enumerate(interaction_cols[i+batch_start+1:], i+batch_start+1):
                                if new_features_count >= target_new_features:
                                    break
                                
                                # Addition
                                series_list.append(
                                    (pl.col(col1) + pl.col(col2)).alias(f"cpu_add_{col1}_{col2}")
                                )
                                new_features_count += 1
                                
                                if new_features_count >= target_new_features:
                                    break
                                
                                # Multiplication
                                series_list.append(
                                    (pl.col(col1) * pl.col(col2)).alias(f"cpu_mul_{col1}_{col2}")
                                )
                                new_features_count += 1
                        
                        # Add all series in one batch
                        if series_list:
                            df_expanded = df_expanded.with_columns(series_list)
                            
                            # Check memory after each batch
                            self.manage_memory(threshold_gb=500)
                
                # CPU Mathematical transformations
                if new_features_count < target_new_features:
                    transform_cols = numeric_cols[:min(50, len(numeric_cols))]
                    
                    # Process in batches to manage memory
                    batch_size = 10
                    for batch_start in range(0, len(transform_cols), batch_size):
                        if new_features_count >= target_new_features:
                            break
                        
                        batch_end = min(batch_start + batch_size, len(transform_cols))
                        batch_cols = transform_cols[batch_start:batch_end]
                        
                        series_list = []
                        for col in batch_cols:
                            if new_features_count >= target_new_features:
                                break
                            
                            # Square
                            series_list.append(
                                (pl.col(col) ** 2).alias(f"cpu_sq_{col}")
                            )
                            new_features_count += 1
                        
                        # Add all series in one batch
                        if series_list:
                            df_expanded = df_expanded.with_columns(series_list)
                            
                            # Check memory after each batch
                            self.manage_memory(threshold_gb=500)
            
            # 3. Statistical aggregations by symbol (if symbol column exists)
            if new_features_count < target_new_features and 'symbol' in df.columns:
                logger.info("  📈 Generating symbol-based aggregations...")
                agg_cols = numeric_cols[:min(50, len(numeric_cols))]
                
                # Calculate symbol-level statistics
                symbol_stats = df.group_by('symbol').agg([
                    pl.col(col).mean().alias(f"sym_mean_{col}") for col in agg_cols[:10]
                ] + [
                    pl.col(col).std().alias(f"sym_std_{col}") for col in agg_cols[:10]
                ])
                
                # Join back to main dataframe
                df_expanded = df_expanded.join(symbol_stats, on='symbol', how='left')
                new_features_count += len(symbol_stats.columns) - 1  # -1 for symbol column
                
                # Clean up memory
                del symbol_stats
                self.manage_memory(threshold_gb=500)
            
            # 4. Rank-based features
            if new_features_count < target_new_features:
                logger.info("  🎯 Generating rank-based features...")
                rank_cols = numeric_cols[:min(20, len(numeric_cols))]
                
                # Process in batches to manage memory
                batch_size = 5
                for batch_start in range(0, len(rank_cols), batch_size):
                    if new_features_count >= target_new_features:
                        break
                    
                    batch_end = min(batch_start + batch_size, len(rank_cols))
                    batch_cols = rank_cols[batch_start:batch_end]
                    
                    series_list = []
                    for col in batch_cols:
                        if new_features_count >= target_new_features:
                            break
                        
                        series_list.append(
                            pl.col(col).rank().alias(f"rank_{col}")
                        )
                        new_features_count += 1
                    
                    # Add all series in one batch
                    if series_list:
                        df_expanded = df_expanded.with_columns(series_list)
                        
                        # Check memory after each batch
                        self.manage_memory(threshold_gb=500)
            
            elapsed = time.time() - start_time
            final_new_features = df_expanded.width - df.width
            
            # Check for random features in final dataframe
            random_cols = [col for col in df_expanded.columns if col.startswith("random_baseline_")]
            if random_cols:
                logger.info(f"  🎲 {len(random_cols)} random baseline features included")
            
            logger.info(f"✅ Generated {final_new_features} new features in {elapsed:.1f}s")
            logger.info(f"  📊 New shape: {df_expanded.shape}")
            logger.info(f"  💾 Memory: {self.get_memory_usage_gb():.1f}GB")
            
            return df_expanded
            
        except Exception as e:
            logger.error(f"❌ Fast feature generation failed in iteration {iteration}: {e}")
            logger.warning("🔄 Returning original dataset")
            return df

    def reduce_features_iteration(self, df: pl.DataFrame, iteration: int, preserve_random: bool = False, 
                                batch_mode: bool = False) -> pl.DataFrame:
        """
        Apply conservative reduction for one iteration
        
        Args:
            df: Input DataFrame
            iteration: Current iteration number
            preserve_random: Whether to preserve random baseline features
            batch_mode: Whether running in batch mode (affects reduction strategy)
        """
        logger.info(f"🛡️  ITERATION {iteration}: CONSERVATIVE REDUCTION")
        logger.info(f"  📊 Input shape: {df.shape}")
        logger.info(f"  🎯 Max reduction: {self.max_reduction*100:.0f}%")
        
        # Check for random baseline features
        random_cols = [col for col in df.columns if col.startswith("random_baseline_")]
        if random_cols:
            if preserve_random:
                logger.info(f"  🎲 Found {len(random_cols)} random baseline features to preserve")
            else:
                logger.info(f"  🎲 Found {len(random_cols)} random baseline features (not preserved in intermediate steps)")
        
        start_time = time.time()
        
        try:
            # Adjust reduction parameters for batch mode
            if batch_mode:
                # In batch mode, use more aggressive reduction per batch (70% instead of default 45%)
                logger.info(f"  🔄 Batch mode: using more aggressive reduction")
                max_reduction = 0.7  # 70% reduction in batch mode
            else:
                max_reduction = self.max_reduction
                
            # Apply conservative reduction with random feature preservation if needed
            df_reduced = self.conservative_reducer.perform_conservative_reduction(
                df, 
                preserve_random_baselines=preserve_random,
                max_reduction=max_reduction
            )
            
            elapsed = time.time() - start_time
            reduction_pct = (1 - df_reduced.width / df.width) * 100
            
            logger.info(f"✅ Reduction completed in {elapsed:.1f}s")
            logger.info(f"  📊 Shape: {df.shape} → {df_reduced.shape}")
            logger.info(f"  📉 Reduction: {reduction_pct:.1f}%")
            logger.info(f"  💾 Memory: {self.get_memory_usage_gb():.1f}GB")
            
            # Check if random baseline features were preserved
            if preserve_random and random_cols:
                preserved_random = [col for col in df_reduced.columns if col.startswith("random_baseline_")]
                logger.info(f"  🎲 Preserved {len(preserved_random)}/{len(random_cols)} random baseline features")
            
            # Record iteration results
            iteration_stats = {
                'iteration': iteration,
                'input_shape': df.shape,
                'output_shape': df_reduced.shape,
                'reduction_pct': reduction_pct,
                'elapsed_time': elapsed,
                'memory_gb': self.get_memory_usage_gb()
            }
            self.evolution_history.append(iteration_stats)
            
            return df_reduced
            
        except Exception as e:
            logger.error(f"❌ Feature reduction failed in iteration {iteration}: {e}")
            logger.warning("🔄 Returning original dataset")
            return df

    def save_iteration_checkpoint(self, df: pl.DataFrame, iteration: int) -> str:
        """Save checkpoint for current iteration"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_file = self.checkpoint_dir / f"fast_evolution_iter_{iteration}_{timestamp}.parquet"
        
        logger.info(f"💾 Saving iteration {iteration} checkpoint...")
        df.write_parquet(checkpoint_file)
        
        # Save evolution history
        history_file = self.checkpoint_dir / f"fast_evolution_history_{timestamp}.json"
        import json
        with open(history_file, 'w') as f:
            json.dump(self.evolution_history, f, indent=2)
        
        logger.info(f"✅ Checkpoint saved: {checkpoint_file}")
        return str(checkpoint_file)

    def run_fast_evolution(self, input_file: str, output_dir: str) -> Tuple[str, Dict]:
        """Run the fast iterative feature evolution pipeline with batched processing"""
        logger.info("🚀 STARTING FAST ITERATIVE FEATURE EVOLUTION")
        logger.info("=" * 80)
        
        total_start_time = time.time()
        
        # Start with memory cleanup
        self.manage_memory(force_cleanup=True, threshold_gb=400)
        
        # Load initial data
        logger.info("📖 Loading initial dataset...")
        try:
            # Check file size to determine loading strategy
            import os
            file_size_gb = os.path.getsize(input_file) / (1024**3)
            logger.info(f"Input file size: {file_size_gb:.2f} GB")
            
            # For large files, use low_memory=True and potentially sample
            if file_size_gb > 5.0:
                logger.info("Large file detected - using memory-efficient loading strategy")
                
                # Check system memory for adaptive sampling
                try:
                    import psutil
                    system_memory = psutil.virtual_memory()
                    system_available_gb = system_memory.available / (1024**3)
                    system_used_percent = system_memory.percent
                    
                    logger.info(f"System memory before loading: {system_available_gb:.1f}GB available, {system_used_percent:.1f}% used")
                    
                    # Adapt sample size based on available memory and file size
                    if self.sample_size == 0:  # No explicit sample requested
                        if system_available_gb < 100 or system_used_percent > 80:
                            # Low memory, take a smaller sample
                            adaptive_sample_size = 250000
                            logger.warning(f"Low memory detected ({system_available_gb:.1f}GB free), using adaptive sample size: {adaptive_sample_size}")
                            self.sample_size = adaptive_sample_size
                        elif file_size_gb > 10.0 and system_available_gb < 200:
                            # Large file with moderate memory, take a moderate sample
                            adaptive_sample_size = 500000
                            logger.warning(f"Large file with moderate memory ({system_available_gb:.1f}GB free), using adaptive sample size: {adaptive_sample_size}")
                            self.sample_size = adaptive_sample_size
                except (ImportError, Exception) as e:
                    logger.debug(f"Could not check system memory for adaptive sampling: {e}")
                
                # Use memory-efficient read options
                df_current = pl.read_parquet(
                    input_file,
                    memory_map=True,
                    n_rows=None if self.sample_size == 0 else self.sample_size
                )
                
                # Force dtype conversion to reduce memory
                for col in df_current.columns:
                    if df_current[col].dtype == pl.Float64:
                        df_current = df_current.with_columns(pl.col(col).cast(pl.Float32))
                
                # If using the full dataset but file is very large, take a sample for memory safety
                if self.sample_size == 0 and file_size_gb > 10.0:
                    logger.warning(f"Very large file ({file_size_gb:.2f} GB) and no sample size specified.")
                    logger.warning("Taking a large sample (500k rows) to ensure processing succeeds.")
                    logger.warning("To use the full dataset, specify a larger memory limit with --memory-limit-gb")
                    
                    # Take a large but manageable sample
                    sample_size = 500000
                    df_current = df_current.sample(sample_size)
                    logger.info(f"Sampled {sample_size} rows from {df_current.shape[0]} total rows")
                
            else:
                # Normal loading for smaller files
                df_current = pl.read_parquet(input_file)
                
                # If sample size specified, take a sample
                if self.sample_size > 0:
                    df_current = df_current.sample(min(self.sample_size, df_current.shape[0]))
            
            logger.info(f"✅ Loaded: {df_current.shape}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load {input_file}: {e}")
            logger.error("Trying alternative loading method...")
            
            # Force memory cleanup before trying alternative method
            self.manage_memory(force_cleanup=True, threshold_gb=400)
            
            try:
                # Try a different loading approach with even more memory conservation
                import pyarrow.parquet as pq
                import pyarrow as pa
                
                # First just read metadata to check schema
                parquet_file = pq.ParquetFile(input_file)
                logger.info(f"ParquetFile metadata loaded, num_row_groups: {parquet_file.num_row_groups}")
                
                # Read data in chunks to manage memory
                table_chunks = []
                rows_to_read = self.sample_size if self.sample_size > 0 else None
                
                for i in range(min(3, parquet_file.num_row_groups)):  # Read at most 3 row groups
                    if rows_to_read is not None and rows_to_read <= 0:
                        break
                        
                    # Read a chunk with batched rows
                    chunk = next(parquet_file.iter_batches(batch_size=min(100000, rows_to_read or 100000)))
                    table_chunks.append(pa.Table.from_batches([chunk]))
                    
                    if rows_to_read is not None:
                        rows_to_read -= len(chunk)
                    
                    # Check memory after each chunk
                    self.manage_memory(threshold_gb=500)
                
                # Combine chunks
                table = pa.concat_tables(table_chunks)
                df_current = pl.from_arrow(table)
                
                logger.info(f"✅ Loaded with alternative method: {df_current.shape}")
                
            except Exception as e2:
                logger.error(f"❌ Both loading methods failed: {e2}")
                raise
        
        # Manage memory after loading
        self.manage_memory(threshold_gb=400)
        
        initial_shape = df_current.shape
        
        # Check if we should use batch mode based on data size
        use_batch_mode = df_current.shape[0] > 1000000 or df_current.shape[1] > 5000
        batch_size = 300  # Number of features to add/process at once in batch mode
        
        if use_batch_mode:
            logger.info(f"🔄 Using batch mode for large dataset with batch size {batch_size}")
        
        # Evolution loop
        for iteration in range(1, self.max_iterations + 1):
            logger.info("")
            logger.info(f"🚀 EVOLUTION CYCLE {iteration}/{self.max_iterations}")
            logger.info("-" * 60)
            
            iteration_start = time.time()
            
            # Clear memory before starting iteration
            self.manage_memory(force_cleanup=True, threshold_gb=400)
            
            if use_batch_mode:
                # Batch mode: Add features in small batches, reduce after each batch
                logger.info(f"🔄 Batch mode: processing features in batches of {batch_size}")
                
                # Step 1: Start with original data for this iteration
                df_batch = df_current.clone()
                
                # Calculate total batches to process
                total_batches = max(1, self.features_per_iteration // batch_size)
                logger.info(f"🔄 Processing {total_batches} feature batches")
                
                for batch_num in range(1, total_batches + 1):
                    batch_start = time.time()
                    logger.info(f"🔄 Batch {batch_num}/{total_batches}")
                    
                    # Generate a small batch of features
                    df_with_new_features = self.generate_fast_features(
                        df_batch, 
                        iteration, 
                        max_new_features=batch_size
                    )
                    
                    # Memory cleanup
                    self.manage_memory(threshold_gb=400)
                    
                    # Reduce features after each batch to keep memory usage low
                    # For intermediate batches, don't preserve random features
                    preserve_random = (iteration == self.max_iterations and batch_num == total_batches)
                    
                    df_batch = self.reduce_features_iteration(
                        df_with_new_features, 
                        iteration, 
                        preserve_random,
                        batch_mode=True
                    )
                    
                    # Memory cleanup
                    del df_with_new_features
                    self.manage_memory(force_cleanup=True, threshold_gb=400)
                    
                    batch_elapsed = time.time() - batch_start
                    logger.info(f"✅ Batch {batch_num} completed in {batch_elapsed:.1f}s")
                
                # Save checkpoint after all batches are processed
                checkpoint_file = self.save_iteration_checkpoint(df_batch, iteration)
                
                # Update current dataset for next iteration
                df_current = df_batch
            else:
                # Standard mode: Process all features at once for smaller datasets
                
                # Step 1: Generate fast features
                df_with_features = self.generate_fast_features(df_current, iteration)
                
                # Aggressive memory cleanup after feature generation
                self.manage_memory(force_cleanup=True, threshold_gb=400)
                
                # Step 2: Apply conservative reduction
                # If this is the last iteration, make sure we keep random features
                preserve_random = (iteration == self.max_iterations)
                df_reduced = self.reduce_features_iteration(df_with_features, iteration, preserve_random)
                
                # Memory cleanup after reduction
                del df_with_features
                self.manage_memory(force_cleanup=True, threshold_gb=400)
                
                # Step 3: Save checkpoint
                checkpoint_file = self.save_iteration_checkpoint(df_reduced, iteration)
                
                # Update current dataset for next iteration
                df_current = df_reduced
            
            iteration_elapsed = time.time() - iteration_start
            logger.info(f"⏱️  Iteration {iteration} completed in {iteration_elapsed:.1f}s")
            
            # Check memory usage - if it's high, do another cleanup
            memory_gb = self.get_memory_usage_gb()
            if memory_gb > self.memory_limit_gb * 0.8:
                logger.warning(f"⚠️  High memory usage after iteration: {memory_gb:.1f}GB")
                self.manage_memory(force_cleanup=True, threshold_gb=400)
        
        # Save final result
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_output = os.path.join(output_dir, f"fast_evolved_features_{timestamp}.parquet")
        
        logger.info("")
        logger.info("💾 Saving final evolved features...")
        df_current.write_parquet(final_output)
        
        total_elapsed = time.time() - total_start_time
        
        # Create evolution summary
        evolution_summary = {
            'initial_shape': initial_shape,
            'final_shape': df_current.shape,
            'total_iterations': self.max_iterations,
            'total_time': total_elapsed,
            'final_memory_gb': self.get_memory_usage_gb(),
            'evolution_history': self.evolution_history,
            'features_per_iteration': self.features_per_iteration,
            'max_reduction': self.max_reduction,
            'output_file': final_output
        }
        
        # Log final summary
        logger.info("")
        logger.info("🎉 FAST ITERATIVE EVOLUTION COMPLETED!")
        logger.info("=" * 80)
        logger.info(f"📊 Initial shape: {initial_shape}")
        logger.info(f"📊 Final shape: {df_current.shape}")
        logger.info(f"🔄 Iterations: {self.max_iterations}")
        logger.info(f"⏱️  Total time: {total_elapsed:.1f}s")
        logger.info(f"💾 Final memory: {self.get_memory_usage_gb():.1f}GB")
        logger.info(f"📁 Output: {final_output}")
        
        return final_output, evolution_summary

def main():
    parser = argparse.ArgumentParser(description="Fast Iterative Feature Evolution Pipeline")
    parser.add_argument("--input-file", required=True, help="Input parquet file")
    parser.add_argument("--output-dir", default="/media/knight2/EDB/numer_crypto_temp/data/features", 
                       help="Output directory")
    parser.add_argument("--checkpoint-dir", default="/media/knight2/EDB/numer_crypto_temp/fast_evolution_checkpoints",
                       help="Checkpoint directory")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum evolution iterations")
    parser.add_argument("--features-per-iteration", type=int, default=10000, 
                       help="Target features to generate per iteration")
    parser.add_argument("--max-reduction", type=float, default=0.4, 
                       help="Maximum reduction ratio (0.4 = max 40% reduction)")
    parser.add_argument("--min-features", type=int, default=2000, 
                       help="Minimum features to keep per iteration")
    parser.add_argument("--memory-limit-gb", type=float, default=120.0, help="Memory limit in GB")
    parser.add_argument("--sample-size", type=int, default=0, 
                       help="Sample size for evaluation (0 = use full dataset, no subsampling)")
    parser.add_argument("--use-gpu", action="store_true", default=True, help="Use GPU acceleration if available")
    parser.add_argument("--no-gpu", dest="use_gpu", action="store_false", help="Disable GPU acceleration")
    # Add support for additional arguments that might be passed
    parser.add_argument("--use-random-baselines", action="store_true", help="Use random baselines for feature generation")
    parser.add_argument("--no-random-baselines", dest="use_random_baselines", action="store_false", help="Disable random baselines")
    parser.add_argument("--preserve-random-baselines", action="store_true", help="Preserve random baselines between iterations")
    parser.add_argument("--no-examples", action="store_true", help="Don't use example-based selection")
    parser.add_argument("--batch-size", type=int, default=5000, help="Batch size for processing")
    parser.add_argument("--threads", type=int, default=48, help="Number of threads to use")
    
    args = parser.parse_args()
    
    # Initialize fast evolution
    evolution = FastIterativeEvolution(
        checkpoint_dir=args.checkpoint_dir,
        max_iterations=args.max_iterations,
        features_per_iteration=args.features_per_iteration,
        max_reduction=args.max_reduction,
        min_features_per_iteration=args.min_features,
        memory_limit_gb=args.memory_limit_gb,
        sample_size=args.sample_size,
        use_gpu=args.use_gpu,
        use_random_baselines=args.use_random_baselines
    )
    
    # Run evolution
    final_output, summary = evolution.run_fast_evolution(
        input_file=args.input_file,
        output_dir=args.output_dir
    )
    
    logger.info(f"🎉 Fast evolution complete! Final output: {final_output}")

if __name__ == "__main__":
    main()