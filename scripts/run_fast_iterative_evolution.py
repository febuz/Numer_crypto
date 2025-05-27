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
                 sample_size: int = 100000,
                 use_gpu: bool = True):
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
        
        # Check GPU availability
        self.gpu_available = self._check_gpu_availability()
        
        # Initialize conservative reducer (it has built-in GPU support)
        self.conservative_reducer = ConservativeReducer(
            checkpoint_dir=str(self.checkpoint_dir / "reducer_checkpoints"),
            max_reduction=max_reduction,
            min_features=min_features_per_iteration,
            memory_limit_gb=memory_limit_gb,
            sample_size=sample_size
        )
        
        self.evolution_history = []
        
        logger.info("🚀 FAST ITERATIVE FEATURE EVOLUTION INITIALIZED")
        logger.info(f"  🔄 Max iterations: {max_iterations}")
        logger.info(f"  🎯 Features per iteration: {features_per_iteration}")
        logger.info(f"  🛡️  Max reduction: {max_reduction*100:.0f}%")
        logger.info(f"  🔢 Min features per iteration: {min_features_per_iteration}")
        logger.info(f"  💾 Memory limit: {memory_limit_gb}GB")

    def get_memory_usage_gb(self) -> float:
        """Get current memory usage in GB"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024 / 1024
        except ImportError:
            return 0.0

    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available"""
        if not self.use_gpu:
            return False
        
        try:
            import cupy as cp
            # Test GPU availability
            cp.array([1, 2, 3])
            logger.info("🚀 GPU acceleration available (CuPy)")
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
            
            # Initialize GPU math accelerator
            math_accelerator = GPUMathAccelerator()
            
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
            
            # Initialize GPU math accelerator
            math_accelerator = GPUMathAccelerator()
            
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

    def generate_fast_features(self, df: pl.DataFrame, iteration: int) -> pl.DataFrame:
        """Generate features using fast mathematical operations"""
        logger.info(f"🚀 ITERATION {iteration}: FAST FEATURE GENERATION")
        logger.info(f"  📊 Input shape: {df.shape}")
        logger.info(f"  💾 Memory: {self.get_memory_usage_gb():.1f}GB")
        
        start_time = time.time()
        
        try:
            # Get numeric columns excluding metadata
            excluded_cols = ['era', 'target', 'symbol', 'date', 'asset', '__index_level_0__']
            numeric_cols = [col for col in df.columns 
                          if col not in excluded_cols and df[col].dtype in [
                              pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int16, pl.Int8
                          ]]
            
            logger.info(f"  🔢 Working with {len(numeric_cols)} numeric columns")
            logger.info(f"  🚀 GPU Available: {self.gpu_available}")
            
            # Start with original data
            df_expanded = df.clone()
            new_features_count = 0
            target_new_features = min(self.features_per_iteration, len(numeric_cols) * 20)
            
            # GPU-accelerated feature generation
            if self.gpu_available and len(numeric_cols) >= 2:
                logger.info("  🚀 Using GPU-accelerated feature generation...")
                
                # Convert numeric columns to numpy for GPU processing
                numeric_data = df.select(numeric_cols).to_numpy().astype(np.float32)
                
                # 1. GPU Cross-feature interactions
                interaction_target = min(target_new_features // 2, 2000)
                gpu_interactions, interaction_names = self._gpu_feature_interactions(
                    numeric_data, numeric_cols, interaction_target
                )
                
                # 2. GPU Mathematical transformations
                remaining_features = target_new_features - len(interaction_names)
                transform_target = min(remaining_features, 1000)
                gpu_transforms, transform_names = self._gpu_math_transforms(
                    numeric_data, numeric_cols, transform_target
                )
                
                # Combine GPU-generated features
                if gpu_interactions.shape[1] > 0:
                    # Convert GPU features back to Polars DataFrame
                    for i, col_name in enumerate(interaction_names):
                        df_expanded = df_expanded.with_columns([
                            pl.Series(col_name, gpu_interactions[:, i])
                        ])
                        new_features_count += 1
                
                if gpu_transforms.shape[1] > 0:
                    for i, col_name in enumerate(transform_names):
                        df_expanded = df_expanded.with_columns([
                            pl.Series(col_name, gpu_transforms[:, i])
                        ])
                        new_features_count += 1
                
                logger.info(f"  ✅ GPU generated {new_features_count} features")
            
            # Fallback to CPU if GPU not available or for additional features
            if new_features_count < target_new_features:
                remaining_target = target_new_features - new_features_count
                logger.info(f"  💻 CPU fallback for {remaining_target} additional features...")
                
                # CPU Cross-feature interactions
                if len(numeric_cols) >= 2:
                    interaction_cols = numeric_cols[:min(30, len(numeric_cols))]
                    
                    for i, col1 in enumerate(interaction_cols):
                        if new_features_count >= target_new_features:
                            break
                        for j, col2 in enumerate(interaction_cols[i+1:], i+1):
                            if new_features_count >= target_new_features:
                                break
                            
                            # Addition
                            df_expanded = df_expanded.with_columns([
                                (pl.col(col1) + pl.col(col2)).alias(f"cpu_add_{col1}_{col2}")
                            ])
                            new_features_count += 1
                            
                            if new_features_count >= target_new_features:
                                break
                            
                            # Multiplication
                            df_expanded = df_expanded.with_columns([
                                (pl.col(col1) * pl.col(col2)).alias(f"cpu_mul_{col1}_{col2}")
                            ])
                            new_features_count += 1
                
                # CPU Mathematical transformations
                if new_features_count < target_new_features:
                    transform_cols = numeric_cols[:min(50, len(numeric_cols))]
                    
                    for col in transform_cols:
                        if new_features_count >= target_new_features:
                            break
                        
                        # Square
                        df_expanded = df_expanded.with_columns([
                            (pl.col(col) ** 2).alias(f"cpu_sq_{col}")
                        ])
                        new_features_count += 1
            
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
            
            # 4. Rank-based features
            if new_features_count < target_new_features:
                logger.info("  🎯 Generating rank-based features...")
                rank_cols = numeric_cols[:min(20, len(numeric_cols))]
                
                for col in rank_cols:
                    if new_features_count >= target_new_features:
                        break
                    
                    df_expanded = df_expanded.with_columns([
                        pl.col(col).rank().alias(f"rank_{col}")
                    ])
                    new_features_count += 1
            
            elapsed = time.time() - start_time
            final_new_features = df_expanded.width - df.width
            
            logger.info(f"✅ Generated {final_new_features} new features in {elapsed:.1f}s")
            logger.info(f"  📊 New shape: {df_expanded.shape}")
            logger.info(f"  💾 Memory: {self.get_memory_usage_gb():.1f}GB")
            
            return df_expanded
            
        except Exception as e:
            logger.error(f"❌ Fast feature generation failed in iteration {iteration}: {e}")
            logger.warning("🔄 Returning original dataset")
            return df

    def reduce_features_iteration(self, df: pl.DataFrame, iteration: int) -> pl.DataFrame:
        """Apply conservative reduction for one iteration"""
        logger.info(f"🛡️  ITERATION {iteration}: CONSERVATIVE REDUCTION")
        logger.info(f"  📊 Input shape: {df.shape}")
        logger.info(f"  🎯 Max reduction: {self.max_reduction*100:.0f}%")
        
        start_time = time.time()
        
        try:
            # Apply conservative reduction
            df_reduced = self.conservative_reducer.perform_conservative_reduction(df)
            
            elapsed = time.time() - start_time
            reduction_pct = (1 - df_reduced.width / df.width) * 100
            
            logger.info(f"✅ Reduction completed in {elapsed:.1f}s")
            logger.info(f"  📊 Shape: {df.shape} → {df_reduced.shape}")
            logger.info(f"  📉 Reduction: {reduction_pct:.1f}%")
            logger.info(f"  💾 Memory: {self.get_memory_usage_gb():.1f}GB")
            
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
        """Run the fast iterative feature evolution pipeline"""
        logger.info("🚀 STARTING FAST ITERATIVE FEATURE EVOLUTION")
        logger.info("=" * 80)
        
        total_start_time = time.time()
        
        # Load initial data
        logger.info("📖 Loading initial dataset...")
        try:
            df_current = pl.read_parquet(input_file)
            logger.info(f"✅ Loaded: {df_current.shape}")
        except Exception as e:
            logger.error(f"❌ Failed to load {input_file}: {e}")
            raise
        
        initial_shape = df_current.shape
        
        # Evolution loop
        for iteration in range(1, self.max_iterations + 1):
            logger.info("")
            logger.info(f"🚀 EVOLUTION CYCLE {iteration}/{self.max_iterations}")
            logger.info("-" * 60)
            
            iteration_start = time.time()
            
            # Step 1: Generate fast features
            df_with_features = self.generate_fast_features(df_current, iteration)
            
            # Memory cleanup
            gc.collect()
            
            # Step 2: Apply conservative reduction
            df_reduced = self.reduce_features_iteration(df_with_features, iteration)
            
            # Memory cleanup
            del df_with_features
            gc.collect()
            
            # Step 3: Save checkpoint
            checkpoint_file = self.save_iteration_checkpoint(df_reduced, iteration)
            
            # Update current dataset for next iteration
            df_current = df_reduced
            
            iteration_elapsed = time.time() - iteration_start
            logger.info(f"⏱️  Iteration {iteration} completed in {iteration_elapsed:.1f}s")
            
            # Check memory usage
            memory_gb = self.get_memory_usage_gb()
            if memory_gb > self.memory_limit_gb * 0.8:
                logger.warning(f"⚠️  High memory usage: {memory_gb:.1f}GB")
                gc.collect()
        
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
    parser.add_argument("--max-iterations", type=int, default=5, help="Maximum evolution iterations")
    parser.add_argument("--features-per-iteration", type=int, default=5000, 
                       help="Target features to generate per iteration")
    parser.add_argument("--max-reduction", type=float, default=0.45, 
                       help="Maximum reduction ratio (0.45 = max 45% reduction)")
    parser.add_argument("--min-features", type=int, default=1000, 
                       help="Minimum features to keep per iteration")
    parser.add_argument("--memory-limit-gb", type=float, default=120.0, help="Memory limit in GB")
    parser.add_argument("--sample-size", type=int, default=100000, help="Sample size for evaluation")
    parser.add_argument("--use-gpu", action="store_true", default=True, help="Use GPU acceleration if available")
    parser.add_argument("--no-gpu", dest="use_gpu", action="store_false", help="Disable GPU acceleration")
    
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
        use_gpu=args.use_gpu
    )
    
    # Run evolution
    final_output, summary = evolution.run_fast_evolution(
        input_file=args.input_file,
        output_dir=args.output_dir
    )
    
    logger.info(f"🎉 Fast evolution complete! Final output: {final_output}")

if __name__ == "__main__":
    main()