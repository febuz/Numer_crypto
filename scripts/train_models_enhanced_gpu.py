#!/usr/bin/env python3
"""
Enhanced Multi-GPU Model Training for Numerai Crypto

This script provides maximum GPU utilization with GPU-accelerated models only:
- Azure Synapse LightGBM GPU (3 variants)
- XGBoost GPU (3 variants) 
- CatBoost GPU (3 variants)
- PyTorch Neural Networks (3 variants)
"""

import os
import sys
import time
import logging
import argparse
import multiprocessing as mp
import numpy as np
import pickle
import torch
import torch.nn as nn
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from datetime import datetime, date
import glob

# Set multiprocessing start method
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()

# Configure GPU settings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["OMP_NUM_THREADS"] = "4"

# Configure Azure Synapse LightGBM by default
os.environ["USE_AZURE_SYNAPSE_LIGHTGBM"] = "1"
os.environ["LIGHTGBM_SYNAPSE_MODE"] = "1"
os.environ["LIGHTGBM_USE_SYNAPSE"] = "1"

# Default paths
GPU_FEATURES_FILE = "/media/knight2/EDB/numer_crypto_temp/data/features/gpu_features.parquet"
MODELS_DIR = "/media/knight2/EDB/numer_crypto_temp/models"

def check_models_exist_today(model_type, models_dir=MODELS_DIR):
    """Check if models of the specified type already exist from today"""
    today_str = date.today().strftime("%Y%m%d")
    
    # Pattern to match model files created today
    if model_type == 'all':
        # Check for any model type
        patterns = [
            f"*lightgbm*gpu*{today_str}*.pkl",
            f"*xgboost*gpu*{today_str}*.pkl", 
            f"*catboost*gpu*{today_str}*.pkl",
            f"*pytorch*gpu*{today_str}*.pkl",
        ]
    else:
        # Check for specific model type
        patterns = [f"*{model_type}*gpu*{today_str}*.pkl"]
    
    existing_models = []
    for pattern in patterns:
        search_path = os.path.join(models_dir, pattern)
        matching_files = glob.glob(search_path)
        existing_models.extend(matching_files)
    
    if existing_models:
        logging.info(f"Found {len(existing_models)} existing GPU models from today for type '{model_type}':")
        for model in existing_models:
            logging.info(f"  - {os.path.basename(model)}")
        return True
    else:
        logging.info(f"No existing GPU models from today found for type '{model_type}'")
        return False

def load_and_prepare_data():
    """Load and prepare data for training"""
    import polars as pl
    
    # Find the latest feature file in the features directory
    features_dir = "/media/knight2/EDB/numer_crypto_temp/data/features"
    
    # Look for feature files in order of preference
    feature_patterns = [
        "fast_evolved_features_*.parquet",      # Fast iterative evolution output (NEWEST)
        "evolved_features_*.parquet",           # Iterative evolution output
        "conservative_reduced_*.parquet",       # Conservative reducer output
        "smart_gpu_reduced_*.parquet",          # Smart GPU reducer output
        "ultra_fast_reduced_*.parquet",         # Legacy ultra-fast reducer
        "features_*reduced*.parquet", 
        "gpu_features.parquet",
        "fast_cpu_features.parquet",
        "*.parquet"
    ]
    
    feature_file = None
    for pattern in feature_patterns:
        feature_files = glob.glob(os.path.join(features_dir, pattern))
        if feature_files:
            # Get the most recent file
            feature_file = max(feature_files, key=os.path.getmtime)
            break
    
    if not feature_file or not os.path.exists(feature_file):
        logger.error(f"No suitable feature file found in {features_dir}")
        logger.info("Available files:")
        for f in glob.glob(os.path.join(features_dir, "*.parquet")):
            logger.info(f"  {f}")
        return None, None
    
    logger.info(f"Loading data from {feature_file}")
    try:
        df = pl.read_parquet(feature_file)
    except Exception as e:
        logger.error(f"Error loading feature file {feature_file}: {e}")
        return None, None
    logger.info(f"Loaded data with shape {df.shape}")
    
    # Prepare features and target
    if 'target' not in df.columns:
        logger.error("Target column 'target' not found in data")
        return None, None
    
    # Get numeric columns and exclude non-feature columns
    excluded_cols = ['target', 'Symbol', 'symbol', 'Prediction', 'prediction', 'date', 'era', 'id', 'asset', '__index_level_0__']
    
    numeric_cols = []
    for col in df.columns:
        if col not in excluded_cols and df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int16, pl.Int8, pl.UInt32, pl.UInt16, pl.UInt8]:
            numeric_cols.append(col)
    
    if not numeric_cols:
        logger.error("No numeric feature columns found")
        return None, None
    
    # Prepare features
    X_df = df.select(numeric_cols).fill_null(0)
    y_series = df.select('target').fill_null(0)['target']
    
    # Convert to numpy arrays
    X = X_df.to_numpy().astype(np.float32)
    y = y_series.to_numpy().astype(np.float32)
    
    logger.info(f"Prepared data with {X.shape[1]} features")
    return X, y

class SimpleNeuralNet(nn.Module):
    """Simple feedforward neural network for regression"""
    def __init__(self, input_size, hidden_sizes=[512, 256, 128], dropout_rate=0.3):
        super(SimpleNeuralNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()

def train_pytorch_worker(gpu_id, X_train, y_train, model_name, output_dir):
    """Train PyTorch neural network on specific GPU"""
    # Set GPU for this worker
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Set memory limit
    # Get memory limit from environment or use default
    memory_limit_gb = int(os.environ.get("GPU_MEMORY_LIMIT", "24"))
    reserved_memory_bytes = int(memory_limit_gb * 0.9 * 1024 * 1024 * 1024)  # 90% of memory limit in bytes
    
    # Configure PyTorch to limit memory usage
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of available memory
            
            # Set other memory-saving settings
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:512"
    except:
        pass
    
    logger.info(f"Worker {gpu_id} training PyTorch model: {model_name} with memory limit {memory_limit_gb}GB")
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"GPU {gpu_id} using device: {device}")
        
        # Split data
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42 + gpu_id
        )
        
        # Scale features for neural network
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_split)
        X_val_scaled = scaler.transform(X_val)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
        y_train_tensor = torch.FloatTensor(y_train_split).to(device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
        y_val_tensor = torch.FloatTensor(y_val).to(device)
        
        # Create model with different architectures per GPU
        if gpu_id == 0:
            model = SimpleNeuralNet(X_train.shape[1], [512, 256, 128], 0.3)
        elif gpu_id == 1:
            model = SimpleNeuralNet(X_train.shape[1], [256, 128, 64], 0.4)
        else:  # gpu_id == 2
            model = SimpleNeuralNet(X_train.shape[1], [1024, 512, 256], 0.2)
        
        model = model.to(device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        logger.info(f"GPU {gpu_id} training neural network with architecture: {[layer for layer in model.network if isinstance(layer, nn.Linear)]}")
        
        start_time = time.time()
        
        # Training loop
        model.train()
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(200):  # Max epochs
            # Training
            optimizer.zero_grad()
            train_pred = model(X_train_tensor)
            train_loss = criterion(train_pred, y_train_tensor)
            train_loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_tensor)
                val_loss = criterion(val_pred, y_val_tensor)
            
            scheduler.step(val_loss)
            
            # Early stopping
            val_loss_value = val_loss.item()
            if val_loss_value < best_val_loss:
                best_val_loss = val_loss_value
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"GPU {gpu_id} early stopping at epoch {epoch}")
                break
                
            model.train()
        
        training_time = time.time() - start_time
        logger.info(f"GPU {gpu_id} completed PyTorch training in {training_time:.2f} seconds")
        
        # Save model (CPU version for compatibility)
        model_cpu = model.cpu()
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f"{model_name}_gpu{gpu_id}.pkl")
        
        # Save both model and scaler
        torch.save({
            'model': model_cpu,
            'scaler': scaler,
            'model_state_dict': model_cpu.state_dict()
        }, model_path)
        
        logger.info(f"GPU {gpu_id} saved PyTorch model to {model_path}")
        
        return {
            'gpu_id': gpu_id,
            'model_name': model_name,
            'model_path': model_path,
            'training_time': training_time,
            'best_val_loss': best_val_loss,
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"GPU {gpu_id} PyTorch training failed: {e}")
        return {
            'gpu_id': gpu_id,
            'model_name': model_name,
            'status': 'failed',
            'error': str(e)
        }

def train_catboost_worker(gpu_id, X_train, y_train, model_name, output_dir):
    """Train CatBoost model on specific GPU"""
    # Set GPU for this worker
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Ensure subprocess inherits correct Python environment
    import sys
    venv_path = '/media/knight2/EDB/repos/Numer_crypto/numer_crypto_env/lib/python3.12/site-packages'
    if venv_path not in sys.path:
        sys.path.insert(0, venv_path)
    
    # Get memory limit from environment or use default
    memory_limit_gb = int(os.environ.get("GPU_MEMORY_LIMIT", "24"))
    
    # CatBoost will use memory percentage based on this value
    gpu_ram_part = 0.9  # Use 90% of available memory
    
    logger.info(f"Worker {gpu_id} training CatBoost model: {model_name} with memory limit {memory_limit_gb}GB")
    
    try:
        # Ensure we can import CatBoost
        try:
            import catboost as cb
        except ImportError as e:
            logger.error(f"GPU {gpu_id} cannot import CatBoost: {e}")
            logger.error(f"GPU {gpu_id} Python path: {sys.path[:3]}...")
            return {
                'gpu_id': gpu_id,
                'model_name': model_name,
                'status': 'failed',
                'error': f'CatBoost import failed: {e}'
            }
        
        from sklearn.model_selection import train_test_split
        
        # Split data
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42 + gpu_id
        )
        
        # GPU-optimized parameters with variation per GPU for best results
        base_params = {
            'task_type': 'GPU',
            'gpu_ram_part': gpu_ram_part,  # Use specified percentage of GPU memory
            'random_seed': 42 + gpu_id,
            'verbose': False,
            'allow_writing_files': False,
            'boosting_type': 'Ordered',  # More accurate boosting type
            'bootstrap_type': 'Bayesian',  # Better regularization
            'early_stopping_rounds': 50  # Faster early stopping
        }
        
        # Vary parameters by GPU for best prediction quality
        if gpu_id == 0:
            base_params.update({
                'iterations': 2000,  # More iterations for better convergence
                'depth': 8,          # Deeper trees for better signal capture
                'learning_rate': 0.03,  # Slower learning rate for better generalization
                'l2_leaf_reg': 2,    # Less regularization for stronger patterns
                'leaf_estimation_iterations': 10  # More accurate leaf values
            })
        elif gpu_id == 1:
            base_params.update({
                'iterations': 1500,
                'depth': 10,         # Even deeper trees
                'learning_rate': 0.05,
                'l2_leaf_reg': 3,
                'rsm': 0.8,          # Random subspace method for better generalization
                'leaf_estimation_method': 'Newton'  # More accurate leaf estimation
            })
        else:  # gpu_id == 2
            base_params.update({
                'iterations': 1000,  # Faster model
                'depth': 6,
                'learning_rate': 0.08,
                'l2_leaf_reg': 4,
                'one_hot_max_size': 100,  # Better handling of categorical features
                'grow_policy': 'Lossguide'  # Different tree growth policy
            })
        
        logger.info(f"GPU {gpu_id} using device: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        logger.info(f"GPU {gpu_id} training with {base_params['iterations']} iterations, depth {base_params['depth']}")
        
        # Create datasets
        train_pool = cb.Pool(X_train_split, y_train_split)
        val_pool = cb.Pool(X_val, y_val)
        
        start_time = time.time()
        
        # Train model
        model = cb.CatBoostRegressor(**base_params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=100,
            use_best_model=True
        )
        
        training_time = time.time() - start_time
        logger.info(f"GPU {gpu_id} completed CatBoost training in {training_time:.2f} seconds")
        
        # Save model
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f"{model_name}_gpu{gpu_id}.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"GPU {gpu_id} saved model to {model_path}")
        
        # Get training metrics
        best_score = model.get_best_score()['validation']['RMSE']
        best_iteration = model.get_best_iteration()
        
        return {
            'gpu_id': gpu_id,
            'model_name': model_name,
            'model_path': model_path,
            'training_time': training_time,
            'best_score': best_score,
            'best_iteration': best_iteration,
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"GPU {gpu_id} CatBoost training failed: {e}")
        return {
            'gpu_id': gpu_id,
            'model_name': model_name,
            'status': 'failed',
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Enhanced Multi-GPU Model Training')
    parser.add_argument('--model-type', type=str, default='all', 
                       choices=['lightgbm', 'xgboost', 'catboost', 'pytorch', 'all'],
                       help='Model type to train')
    parser.add_argument('--gpus', type=str, default='0,1,2',
                       help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--output-dir', type=str, default=MODELS_DIR,
                       help='Output directory for models')
    parser.add_argument('--force-retrain', action='store_true', 
                       help='Force model retraining even if models exist from today')
    parser.add_argument('--use-azure-synapse', action='store_true', default=True,
                       help='Use Azure Synapse LightGBM variant (enabled by default)')
    parser.add_argument('--parallel-jobs', type=int, default=None, 
                       help='Number of parallel training jobs (defaults to number of GPUs)')
    
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    logger.info(f"Using GPUs: {gpu_ids}")
    
    # Set number of parallel jobs
    max_workers = args.parallel_jobs if args.parallel_jobs is not None else len(gpu_ids)
    logger.info(f"Using {max_workers} parallel training jobs")
    
    # Configure Azure Synapse LightGBM (already set at module level)
    if args.use_azure_synapse:
        logger.info("Using Azure Synapse LightGBM for maximum performance")
        os.environ["USE_AZURE_SYNAPSE_LIGHTGBM"] = "1"
        os.environ["LIGHTGBM_SYNAPSE_MODE"] = "1"
        os.environ["LIGHTGBM_USE_SYNAPSE"] = "1"
        
    # Get GPU memory limit from environment variable
    gpu_memory_limit = int(os.environ.get("GPU_MEMORY_LIMIT", "24"))
    logger.info(f"Using GPU memory limit: {gpu_memory_limit}GB")
    
    # Set environment variable for workers
    os.environ["GPU_MEMORY_LIMIT"] = str(gpu_memory_limit)
    
    # Check if models already exist from today (unless force-retrain is specified)
    if not args.force_retrain:
        if check_models_exist_today(args.model_type, args.output_dir):
            logger.info(f"GPU models for '{args.model_type}' already exist from today.")
            logger.info("Skipping training. Use --force-retrain to retrain existing models.")
            return True
    else:
        logger.info("Force retrain specified - will train GPU models even if they exist from today")
    
    # Load and prepare data
    logger.info("Loading and preparing data...")
    X_train, y_train = load_and_prepare_data()
    
    if X_train is None or y_train is None:
        logger.error("Failed to load data")
        return False
    
    logger.info(f"Training with {X_train.shape[0]} samples and {X_train.shape[1]} features")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = []
    
    # Import workers from train_models_multi_gpu (using the updated Synapse-enabled version)
    from train_models_multi_gpu import train_lightgbm_worker, train_xgboost_worker
    
    # Train models based on type
    if args.model_type in ['lightgbm', 'all']:
        logger.info("Starting multi-GPU LightGBM training...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for gpu_id in gpu_ids:
                future = executor.submit(
                    train_lightgbm_worker, gpu_id, X_train, y_train, "lightgbm", args.output_dir
                )
                futures.append(future)
            
            for future in futures:
                result = future.result()
                all_results.append(result)
                if result['status'] == 'success':
                    logger.info(f"✅ LightGBM GPU {result['gpu_id']}: {result['training_time']:.2f}s, RMSE: {result['best_score']:.6f}")
    
    if args.model_type in ['xgboost', 'all']:
        logger.info("Starting multi-GPU XGBoost training...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for gpu_id in gpu_ids:
                future = executor.submit(
                    train_xgboost_worker, gpu_id, X_train, y_train, "xgboost", args.output_dir
                )
                futures.append(future)
            
            for future in futures:
                result = future.result()
                all_results.append(result)
                if result['status'] == 'success':
                    logger.info(f"✅ XGBoost GPU {result['gpu_id']}: {result['training_time']:.2f}s, RMSE: {result['best_score']:.6f}")
    
    if args.model_type in ['catboost', 'all']:
        logger.info("Starting multi-GPU CatBoost training...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for gpu_id in gpu_ids:
                future = executor.submit(
                    train_catboost_worker, gpu_id, X_train, y_train, "catboost", args.output_dir
                )
                futures.append(future)
            
            for future in futures:
                result = future.result()
                all_results.append(result)
                if result['status'] == 'success':
                    logger.info(f"✅ CatBoost GPU {result['gpu_id']}: {result['training_time']:.2f}s, RMSE: {result['best_score']:.6f}")
    
    if args.model_type in ['pytorch', 'all']:
        logger.info("Starting multi-GPU PyTorch training...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for gpu_id in gpu_ids:
                future = executor.submit(
                    train_pytorch_worker, gpu_id, X_train, y_train, "pytorch", args.output_dir
                )
                futures.append(future)
            
            for future in futures:
                result = future.result()
                all_results.append(result)
                if result['status'] == 'success':
                    logger.info(f"✅ PyTorch GPU {result['gpu_id']}: {result['training_time']:.2f}s, Loss: {result['best_val_loss']:.6f}")
    
    # Summary
    successful_models = [r for r in all_results if r['status'] == 'success']
    failed_models = [r for r in all_results if r['status'] == 'failed']
    skipped_models = [r for r in all_results if r['status'] == 'skipped']
    
    logger.info(f"\n=== Enhanced Training Summary ===")
    logger.info(f"✅ Successful models: {len(successful_models)}")
    logger.info(f"❌ Failed models: {len(failed_models)}")
    logger.info(f"⏭️ Skipped models: {len(skipped_models)}")
    
    if successful_models:
        total_time = max([r['training_time'] for r in successful_models])
        avg_time = sum([r['training_time'] for r in successful_models]) / len(successful_models)
        logger.info(f"🕒 Total training time: {total_time:.2f}s (parallel)")
        logger.info(f"🕒 Average training time: {avg_time:.2f}s per model")
        
        logger.info("\nSuccessful models:")
        for result in successful_models:
            logger.info(f"  {result['model_name']}_gpu{result['gpu_id']}: {result['model_path']}")
    
    return len(successful_models) > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)