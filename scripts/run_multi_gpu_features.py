#!/usr/bin/env python3
"""
Multi-GPU Feature Generation Script for Numer_crypto

This script distributes feature generation work across all available GPUs with optimized parallelization
to better utilize 2x Xeon Platinum CPUs and multiple GPUs.
"""

import os
import sys
import time
import logging
import multiprocessing as mp
import numpy as np
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
import argparse
import signal

# Set multiprocessing start method to 'spawn' to avoid CUDA reinitialization errors
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()

# Configure GPU settings properly
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
sys.path.append(repo_dir)

# Import configuration
try:
    from config.settings import FEATURE_GENERATION_CONFIG
    logger.info("Loaded feature generation configuration from config.settings")
except ImportError:
    logger.warning("Could not load config.settings, using default configuration")
    FEATURE_GENERATION_CONFIG = {
        'max_columns': 5000000,
        'window_sizes': [7, 14, 28],
        'lag_periods': [1, 3, 7, 14],
        'ewm_spans': [10, 20],
        'excluded_columns': {
            'date_col': 'date',
            'target_cols': ['target'],
            'id_cols': ['era', 'id', 'symbol'],
        }
    }

def check_gpu_availability():
    """Check if PyTorch can detect GPUs"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"PyTorch detected {device_count} CUDA devices")
            
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                total_memory = torch.cuda.get_device_properties(i).total_memory
                logger.info(f"GPU {i}: {device_name}, {total_memory/(1024**3):.2f} GB memory")
            
            return device_count
        else:
            logger.error("PyTorch is installed but CUDA is not available")
            return 0
    except ImportError:
        logger.error("PyTorch is not installed")
        return 0

def kill_gpu_processes():
    """Kill all processes using GPU resources"""
    logger.info("Killing all processes using GPU resources...")
    
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            pids = [pid.strip() for pid in result.stdout.split('\n') if pid.strip()]
            logger.info(f"Found {len(pids)} processes using GPU resources: {pids}")
            
            for pid in pids:
                try:
                    subprocess.run(["kill", "-9", pid], check=False)
                    logger.info(f"Killed process with PID {pid}")
                except Exception as e:
                    logger.error(f"Failed to kill process {pid}: {e}")
    except Exception as e:
        logger.error(f"Error killing GPU processes: {e}")

def clean_gpu_memory():
    """Clean GPU memory by forcing garbage collection"""
    logger.info("Cleaning GPU memory...")
    
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"Found {device_count} CUDA devices")
            
            for i in range(device_count):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    memory_allocated = torch.cuda.memory_allocated(i)
                    logger.info(f"GPU {i} memory after cleanup: {memory_allocated / (1024**3):.2f} GB allocated")
    except ImportError:
        logger.warning("PyTorch is not installed")

def create_feature_config(max_features=None):
    """Create feature configuration based on config file settings"""
    base_config = {
        "window_sizes": FEATURE_GENERATION_CONFIG['window_sizes'],
        "lag_periods": FEATURE_GENERATION_CONFIG['lag_periods'],
        "ewm_spans": FEATURE_GENERATION_CONFIG['ewm_spans'],
        "date_col": FEATURE_GENERATION_CONFIG['excluded_columns']['date_col'],
        "target_cols": FEATURE_GENERATION_CONFIG['excluded_columns']['target_cols'],
        "id_cols": FEATURE_GENERATION_CONFIG['excluded_columns']['id_cols'],
        "max_columns": FEATURE_GENERATION_CONFIG['max_columns'],
    }
    
    if max_features and max_features < base_config['max_columns']:
        base_config['max_columns'] = max_features
        logger.info(f'⚠️ Max columns limited to {max_features:,} by max_features parameter')
    
    return base_config

def process_symbol_chunk_on_gpu(gpu_id, symbol_chunk, data_path, feature_config, output_dir, chunk_id):
    """Process a chunk of symbols in parallel on specified GPU using threading"""
    # Set environment variables to use only this GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Add signal handler for graceful shutdown
    def signal_handler(signum, frame):
        logger.warning(f"GPU {gpu_id} chunk {chunk_id} received signal {signum}, shutting down gracefully")
        raise KeyboardInterrupt("Received termination signal")
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Import libraries inside the worker
    try:
        import polars as pl
        import torch
        import numpy as np
    except ImportError:
        logger.error(f"GPU {gpu_id} chunk {chunk_id}: Required libraries not installed")
        return {"gpu_id": gpu_id, "chunk_id": chunk_id, "status": "failed", "error": "Required libraries not installed"}
    
    try:
        # Set GPU device
        device = torch.device(f"cuda:0")  # Always use device 0 since CUDA_VISIBLE_DEVICES is set
        torch.cuda.set_device(device)
        device_name = torch.cuda.get_device_name(0)
        
        logger.info(f"GPU {gpu_id} chunk {chunk_id} using device: {device_name}")
        
        # Load data
        logger.info(f"GPU {gpu_id} chunk {chunk_id} loading data from {data_path}")
        df = pl.read_parquet(data_path)
        
        # Load the feature generation configuration
        window_sizes = feature_config["window_sizes"]
        lag_periods = feature_config["lag_periods"]
        ewm_spans = feature_config["ewm_spans"]
        date_col = feature_config["date_col"]
        target_cols = feature_config["target_cols"]
        id_cols = feature_config["id_cols"]
        max_columns = feature_config["max_columns"]
        
        # Get numeric columns
        excluded_cols = set([date_col] + target_cols + id_cols)
        all_columns = df.columns
        numeric_cols = [col for col in all_columns if col not in excluded_cols]
        
        if max_columns and len(numeric_cols) > max_columns:
            numeric_cols = numeric_cols[:max_columns]
            logger.info(f"GPU {gpu_id} chunk {chunk_id} limited to {max_columns} numeric columns")
        
        # Filter data for this symbol chunk
        chunk_data = df.filter(pl.col("symbol").is_in(symbol_chunk))
        
        if chunk_data.height == 0:
            logger.warning(f"GPU {gpu_id} chunk {chunk_id} found no data for symbols {symbol_chunk}")
            return {
                "gpu_id": gpu_id,
                "chunk_id": chunk_id,
                "symbols": symbol_chunk,
                "status": "empty",
                "features_generated": 0,
                "rows_processed": 0,
                "memory_used_gb": 0,
                "device_name": device_name,
                "output_file": None
            }
        
        # Initialize statistics
        total_features_generated = 0
        total_rows_processed = 0
        start_time = time.time()
        
        # Pre-define all expected feature names
        expected_features = set()
        for window in window_sizes:
            for col in numeric_cols:
                expected_features.add(f"{col}_roll_{window}_mean")
        for lag in lag_periods:
            for col in numeric_cols:
                expected_features.add(f"{col}_lag_{lag}")
        for span in ewm_spans:
            for col in numeric_cols:
                expected_features.add(f"{col}_ewm_{span}")
        for col in numeric_cols[:5]:
            expected_features.add(f"{col}_return_1d")
        
        logger.info(f"GPU {gpu_id} chunk {chunk_id} will generate {len(expected_features)} features per symbol")
        
        # Process symbols in parallel using ThreadPoolExecutor
        def process_single_symbol(symbol):
            """Process individual symbol"""
            try:
                symbol_data = chunk_data.filter(pl.col("symbol") == symbol)
                n_rows = symbol_data.height
                
                if n_rows == 0:
                    return None, 0, 0
                
                # Sort by date
                if date_col in symbol_data.columns:
                    symbol_data = symbol_data.sort(date_col)
                
                # Convert numeric features to torch tensors
                symbol_tensors = {}
                for col in numeric_cols:
                    if col in symbol_data.columns:
                        col_data = symbol_data[col].to_numpy()
                        symbol_tensors[col] = torch.tensor(col_data, dtype=torch.float32, device=device)
                
                # Generate features using GPU
                feature_results = {}
                features_generated = 0
                
                # A. Rolling window features
                for window in window_sizes:
                    for col in numeric_cols:
                        if col not in symbol_tensors:
                            continue
                        
                        tensor = symbol_tensors[col]
                        n = len(tensor)
                        
                        if window >= n:
                            continue
                        
                        if n >= window:
                            windows_tensor = tensor.unfold(0, window, 1)
                            rolling_means = torch.mean(windows_tensor, dim=1)
                            
                            means = torch.zeros(n, device=device)
                            for i in range(min(window-1, n)):
                                means[i] = torch.mean(tensor[:i+1])
                            if len(rolling_means) > 0:
                                means[window-1:] = rolling_means
                            
                            feature_results[f"{col}_roll_{window}_mean"] = means.cpu().numpy().astype(np.float32)
                            features_generated += 1
                
                # B. Lag features
                for lag in lag_periods:
                    for col in numeric_cols:
                        if col not in symbol_tensors:
                            continue
                        
                        tensor = symbol_tensors[col]
                        n = len(tensor)
                        
                        if lag >= n:
                            continue
                        
                        lag_values = torch.zeros(n, device=device)
                        lag_values[lag:] = tensor[:-lag]
                        feature_results[f"{col}_lag_{lag}"] = lag_values.cpu().numpy().astype(np.float32)
                        features_generated += 1
                
                # C. Exponential weighted moving averages
                for span in ewm_spans:
                    alpha = 2.0 / (span + 1.0)
                    for col in numeric_cols:
                        if col not in symbol_tensors:
                            continue
                        
                        tensor = symbol_tensors[col]
                        n = len(tensor)
                        
                        if n < 2:
                            continue
                        
                        ewm_values = torch.zeros(n, device=device)
                        ewm_values[0] = tensor[0]
                        
                        for i in range(1, n):
                            ewm_values[i] = alpha * tensor[i] + (1 - alpha) * ewm_values[i-1]
                        
                        feature_results[f"{col}_ewm_{span}"] = ewm_values.cpu().numpy().astype(np.float32)
                        features_generated += 1
                
                # D. Simple return features
                key_cols = [col for col in numeric_cols[:5] if col in symbol_tensors]
                for col in key_cols:
                    tensor = symbol_tensors[col]
                    n = len(tensor)
                    
                    if n < 2:
                        continue
                    
                    returns = torch.zeros(n, device=device)
                    valid_mask = tensor[:-1] != 0
                    returns[1:][valid_mask] = tensor[1:][valid_mask] / tensor[:-1][valid_mask] - 1
                    feature_results[f"{col}_return_1d"] = returns.cpu().numpy().astype(np.float32)
                    features_generated += 1
                
                # Convert to DataFrame
                symbol_result = symbol_data.clone()
                
                # Cast numeric columns to Float32
                numeric_cast_exprs = []
                for col in symbol_result.columns:
                    if symbol_result[col].dtype in [pl.Float64, pl.Float32, pl.Int32, pl.Int64]:
                        numeric_cast_exprs.append(pl.col(col).cast(pl.Float32))
                    else:
                        numeric_cast_exprs.append(pl.col(col))
                
                if numeric_cast_exprs:
                    symbol_result = symbol_result.select(numeric_cast_exprs)
                
                # Add all expected features
                for feat_name in expected_features:
                    if feat_name in feature_results and len(feature_results[feat_name]) == n_rows:
                        feature_values = feature_results[feat_name]
                        if hasattr(feature_values, 'dtype') and feature_values.dtype != np.float32:
                            feature_values = feature_values.astype(np.float32)
                        symbol_result = symbol_result.with_columns(
                            pl.Series(feat_name, feature_values, dtype=pl.Float32)
                        )
                    else:
                        nan_values = np.full(n_rows, np.nan, dtype=np.float32)
                        symbol_result = symbol_result.with_columns(
                            pl.Series(feat_name, nan_values, dtype=pl.Float32)
                        )
                
                return symbol_result, len(expected_features), n_rows
                
            except Exception as e:
                logger.error(f"GPU {gpu_id} chunk {chunk_id} error processing symbol {symbol}: {e}")
                return None, 0, 0
        
        # Process symbols in parallel using threads (for I/O bound operations)
        all_results = []
        max_workers = min(8, len(symbol_chunk))  # Limit threads to avoid overwhelming, leverage 2x Xeon Platinum
        
        logger.info(f"GPU {gpu_id} chunk {chunk_id} processing {len(symbol_chunk)} symbols with {max_workers} threads")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all symbol processing tasks
            future_to_symbol = {
                executor.submit(process_single_symbol, symbol): symbol 
                for symbol in symbol_chunk
            }
            
            # Collect results as they complete
            completed_count = 0
            for future in concurrent.futures.as_completed(future_to_symbol, timeout=1800):
                symbol = future_to_symbol[future]
                try:
                    result, features_gen, rows_proc = future.result()
                    if result is not None:
                        all_results.append(result)
                        total_features_generated += features_gen
                        total_rows_processed += rows_proc
                        completed_count += 1
                        
                        # Log progress periodically
                        if completed_count % 10 == 0 or completed_count == len(symbol_chunk):
                            elapsed = time.time() - start_time
                            logger.info(f"GPU {gpu_id} chunk {chunk_id} completed {completed_count}/{len(symbol_chunk)} symbols in {elapsed:.2f}s")
                        
                except Exception as e:
                    logger.error(f"GPU {gpu_id} chunk {chunk_id} failed to process symbol {symbol}: {e}")
                    completed_count += 1
        
        # Combine results
        if all_results:
            logger.info(f"GPU {gpu_id} chunk {chunk_id} combining {len(all_results)} symbol results")
            combined_result = pl.concat(all_results, how="vertical")
            
            # Save results
            output_file = os.path.join(output_dir, f"gpu_{gpu_id}_chunk_{chunk_id}_features.parquet")
            os.makedirs(output_dir, exist_ok=True)
            combined_result.write_parquet(output_file)
            logger.info(f"GPU {gpu_id} chunk {chunk_id} saved results to {output_file}")
            
            result_status = "success"
        else:
            combined_result = None
            error_msg = f"GPU {gpu_id} chunk {chunk_id} produced no results"
            logger.error(error_msg)
            result_status = "empty"
        
        # Report memory usage
        memory_used = torch.cuda.memory_allocated(0)
        logger.info(f"GPU {gpu_id} chunk {chunk_id} memory used: {memory_used/(1024**3):.2f} GB")
        torch.cuda.empty_cache()
        
        return {
            "gpu_id": gpu_id,
            "chunk_id": chunk_id,
            "status": result_status,
            "symbols_processed": len(symbol_chunk),
            "rows_processed": total_rows_processed,
            "features_generated": total_features_generated,
            "output_file": output_file if result_status == "success" else None,
            "memory_used_gb": memory_used / (1024**3),
            "device_name": device_name
        }
        
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        logger.error(f"GPU {gpu_id} chunk {chunk_id} worker failed: {e}\n{error_msg}")
        return {"gpu_id": gpu_id, "chunk_id": chunk_id, "status": "failed", "error": str(e)}

def run_multi_gpu_feature_generation(data_path, output_dir, gpus=[0, 1, 2], max_features=None):
    """Run optimized feature generation across multiple GPUs with better parallelization"""
    import polars as pl
    
    logger.info(f"Starting optimized multi-GPU feature generation with GPUs {gpus}")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load data to get symbols for distribution
    logger.info("Loading data to determine symbol distribution...")
    try:
        df = pl.read_parquet(data_path)
        logger.info(f"Loaded data with shape {df.shape}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return []
    
    # Get unique symbols
    symbols = df.select("symbol").unique().to_series().to_list()
    n_symbols = len(symbols)
    logger.info(f"Total symbols: {n_symbols}")
    
    # Create smaller chunks for better load balancing
    # Use more chunks than GPUs to enable work stealing
    chunk_size = max(1, n_symbols // (len(gpus) * 4))  # 4x more chunks than GPUs
    symbol_chunks = []
    
    for i in range(0, n_symbols, chunk_size):
        chunk = symbols[i:i + chunk_size]
        symbol_chunks.append(chunk)
    
    logger.info(f"Created {len(symbol_chunks)} symbol chunks with ~{chunk_size} symbols each")
    
    # Distribute chunks across GPUs in round-robin fashion
    gpu_tasks = []
    for chunk_id, chunk in enumerate(symbol_chunks):
        gpu_id = gpus[chunk_id % len(gpus)]
        gpu_tasks.append({
            'gpu_id': gpu_id,
            'chunk_id': chunk_id,
            'symbols': chunk
        })
    
    # Log distribution
    for gpu_id in gpus:
        gpu_chunks = [task for task in gpu_tasks if task['gpu_id'] == gpu_id]
        total_symbols = sum(len(task['symbols']) for task in gpu_chunks)
        logger.info(f"GPU {gpu_id}: {len(gpu_chunks)} chunks, {total_symbols} symbols")
    
    # Create feature configuration
    feature_config = create_feature_config(max_features)
    
    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # Use ProcessPoolExecutor with more workers for better utilization
    max_workers = min(len(gpu_tasks), len(gpus) * 2)  # Allow 2 processes per GPU
    logger.info(f"Using {max_workers} workers for {len(gpu_tasks)} tasks")
    
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = []
        for task in gpu_tasks:
            future = executor.submit(
                process_symbol_chunk_on_gpu,
                task['gpu_id'],
                task['symbols'],
                data_path,
                feature_config,
                output_dir,
                task['chunk_id']
            )
            futures.append((future, task['gpu_id'], task['chunk_id']))
        
        # Collect results with timeout
        for future, gpu_id, chunk_id in futures:
            try:
                result = future.result(timeout=1800)  # 30 minutes timeout per chunk
                results.append(result)
                if result["status"] == "success":
                    logger.info(f"GPU {gpu_id} chunk {chunk_id} completed successfully")
                else:
                    logger.error(f"GPU {gpu_id} chunk {chunk_id} failed: {result.get('error', 'Unknown error')}")
            except concurrent.futures.TimeoutError:
                logger.error(f"GPU {gpu_id} chunk {chunk_id} timed out after 30 minutes")
                results.append({"gpu_id": gpu_id, "chunk_id": chunk_id, "status": "failed", "error": "Timeout"})
            except Exception as e:
                logger.error(f"Error getting results from GPU {gpu_id} chunk {chunk_id}: {e}")
                results.append({"gpu_id": gpu_id, "chunk_id": chunk_id, "status": "failed", "error": str(e)})
    
    return results

def combine_gpu_results(results, output_file):
    """Combine results from multiple GPU chunks into a single file"""
    import polars as pl
    
    logger.info("Combining results from all GPU chunks")
    
    # Collect all successful result files
    result_files = []
    for result in results:
        if result["status"] == "success" and result.get("output_file"):
            result_files.append(result["output_file"])
    
    if not result_files:
        logger.error("No successful results to combine")
        return None
    
    # Load and combine all result files
    try:
        logger.info(f"Combining {len(result_files)} result files")
        dfs = []
        
        for file_path in result_files:
            logger.info(f"Loading {file_path}")
            df = pl.read_parquet(file_path)
            dfs.append(df)
        
        # Combine all DataFrames
        logger.info("Concatenating all results")
        combined_df = pl.concat(dfs, how="vertical")
        
        # Save combined results
        logger.info(f"Saving combined results to {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        combined_df.write_parquet(output_file)
        
        logger.info(f"Successfully combined {len(dfs)} files into {output_file}")
        logger.info(f"Final shape: {combined_df.shape}")
        
        return combined_df
        
    except Exception as e:
        logger.error(f"Failed to combine results: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def verify_gpu_usage(results):
    """Verify that multiple GPUs were used successfully"""
    successful_gpus = set()
    for result in results:
        if result["status"] == "success":
            successful_gpus.add(result["gpu_id"])
    
    logger.info(f"Successfully used {len(successful_gpus)} GPUs: {sorted(successful_gpus)}")
    return len(successful_gpus) > 1

def check_output_files(results, final_output_file):
    """Check that output files exist and have reasonable sizes"""
    logger.info("Checking output files...")
    
    # Check final output file
    if os.path.exists(final_output_file):
        file_size = os.path.getsize(final_output_file)
        logger.info(f"✅ Final output file: {final_output_file} ({file_size:,} bytes)")
    else:
        logger.warning(f"⚠️  Final output file not found: {final_output_file}")
    
    # Check individual GPU output files
    for result in results:
        if result["status"] == "success" and result.get("output_file"):
            gpu_file = result["output_file"]
            if os.path.exists(gpu_file):
                file_size = os.path.getsize(gpu_file)
                logger.info(f"✅ GPU {result['gpu_id']} chunk {result.get('chunk_id', 0)} output file: {gpu_file} ({file_size:,} bytes)")
            else:
                logger.warning(f"⚠️  GPU {result['gpu_id']} chunk {result.get('chunk_id', 0)} output file not found: {gpu_file}")

def main():
    """Main function to run optimized multi-GPU feature generation"""
    parser = argparse.ArgumentParser(description='Generate features using optimized multi-GPU acceleration')
    parser.add_argument('--input-file', type=str, 
                       default="/media/knight2/EDB/numer_crypto_temp/data/processed/crypto_train.parquet",
                       help='Input data file path')
    parser.add_argument('--output-file', type=str, 
                       default="/media/knight2/EDB/numer_crypto_temp/data/features/gpu_features.parquet",
                       help='Output file path')
    parser.add_argument('--temp-dir', type=str,
                       default="/media/knight2/EDB/numer_crypto_temp/data/features/temp",
                       help='Temporary directory for intermediate files')
    parser.add_argument('--max-features', type=int, default=10000, help='Maximum number of features to generate')
    parser.add_argument('--gpus', type=str, default="0,1,2", help='Comma-separated GPU indices to use')
    parser.add_argument('--skip-cleanup', action='store_true', help='Skip killing processes and cleaning memory')
    
    args = parser.parse_args()
    
    # Parse GPUs
    gpus = [int(idx) for idx in args.gpus.split(",")]
    
    logger.info("=== Starting Optimized Multi-GPU Feature Generation ===")
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Output file: {args.output_file}")
    logger.info(f"Using GPUs: {gpus}")
    logger.info(f"Max features: {args.max_features}")
    
    # Record overall start time
    start_time = time.time()
    
    # Check if GPUs are available
    gpu_count = check_gpu_availability()
    if gpu_count == 0:
        logger.error("No GPUs available. Exiting.")
        return 1
    
    # Validate that requested GPUs are available
    if max(gpus) >= gpu_count:
        logger.error(f"Requested GPU {max(gpus)} but only {gpu_count} GPUs available")
        return 1
    
    # Kill GPU processes and clean memory (unless skipped)
    if not args.skip_cleanup:
        kill_gpu_processes()
        clean_gpu_memory()
    else:
        logger.info("Skipping process cleanup and memory cleaning")
    
    # Validate input file exists
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        return 1
    
    # Run optimized feature generation
    logger.info("Starting optimized feature generation...")
    feature_start_time = time.time()
    results = run_multi_gpu_feature_generation(
        data_path=args.input_file,
        output_dir=args.temp_dir,
        gpus=gpus,
        max_features=args.max_features
    )
    feature_time = time.time() - feature_start_time
    logger.info(f"Feature generation completed in {feature_time:.2f} seconds")
    
    # Combine results
    combine_start_time = time.time()
    combined_df = combine_gpu_results(results, args.output_file)
    combine_time = time.time() - combine_start_time
    logger.info(f"Results combination completed in {combine_time:.2f} seconds")
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Verify the results
    success = verify_gpu_usage(results)
    
    # Check output files
    check_output_files(results, args.output_file)
    
    # Print detailed results
    logger.info("\n=== Detailed GPU Results ===")
    for result in sorted(results, key=lambda r: (r["gpu_id"], r.get("chunk_id", 0))):
        if result["status"] == "success":
            logger.info(f"GPU {result['gpu_id']} chunk {result.get('chunk_id', 0)}: "
                       f"{result['features_generated']} features across {result['symbols_processed']} symbols, "
                       f"{result['rows_processed']} rows, used {result['memory_used_gb']:.2f} GB memory")
        else:
            logger.error(f"GPU {result['gpu_id']} chunk {result.get('chunk_id', 0)}: Failed - {result.get('error', 'Unknown error')}")
    
    # Print performance summary
    if combined_df is not None:
        output_shape = combined_df.shape
        
        # Try to get input shape for comparison
        try:
            import polars as pl
            input_df = pl.read_parquet(args.input_file)
            input_shape = input_df.shape
        except:
            input_shape = None
        
        logger.info("\n=== Performance Summary ===")
        if input_shape:
            logger.info(f"Input data: {input_shape[0]:,} rows × {input_shape[1]} columns")
        logger.info(f"Output data: {output_shape[0]:,} rows × {output_shape[1]} columns")
        if input_shape:
            features_added = output_shape[1] - input_shape[1]
            logger.info(f"Features added: {features_added}")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"  - Feature generation: {feature_time:.2f} seconds")
        logger.info(f"  - Result combination: {combine_time:.2f} seconds")
        logger.info(f"Final output: {args.output_file}")
        
        # Performance metrics
        if feature_time > 0:
            rows_per_second = output_shape[0] / feature_time
            logger.info(f"Processing rate: {rows_per_second:,.0f} rows/second")
    
    logger.info("\n=== Summary ===")
    if success:
        logger.info("✅ Optimized multi-GPU feature generation completed successfully")
        return 0
    else:
        logger.error("❌ Optimized multi-GPU feature generation failed")
        return 1

if __name__ == "__main__":
    logger.info("Using 'spawn' multiprocessing start method for CUDA compatibility")
    sys.exit(main())