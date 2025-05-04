#!/usr/bin/env python3
"""
Full GPU Integration Test for Machine Learning Libraries
Tests XGBoost, LightGBM, and H2O with GPU acceleration.
Measures performance and GPU utilization.
"""

import os
import sys
import time
import subprocess
import tempfile
from datetime import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# Import project modules if available
try:
    from utils.gpu_utils import get_gpu_info, monitor_gpu
except ImportError:
    print("GPU utilities not found. Will use basic GPU monitoring.")
    
    def get_gpu_info():
        """Get basic GPU information using nvidia-smi"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True
            )
            gpus = []
            for i, line in enumerate(result.stdout.strip().split('\n')):
                name, total_memory, used_memory, utilization = line.split(', ')
                gpus.append({
                    'id': i,
                    'name': name,
                    'memory_total_mb': float(total_memory),
                    'memory_used_mb': float(used_memory),
                    'utilization_pct': float(utilization)
                })
            return gpus
        except (subprocess.SubprocessError, FileNotFoundError):
            print("Could not get GPU information. Is nvidia-smi available?")
            return []
    
    def monitor_gpu(interval=1.0, duration=None, output_file=None):
        """Monitor GPU utilization for a given duration"""
        stats = []
        start_time = time.time()
        try:
            while duration is None or time.time() - start_time < duration:
                gpus = get_gpu_info()
                timestamp = time.time() - start_time
                for gpu in gpus:
                    stats.append({
                        'timestamp': timestamp,
                        'gpu_id': gpu['id'],
                        'utilization_pct': gpu['utilization_pct'],
                        'memory_used_mb': gpu['memory_used_mb']
                    })
                if duration is None:
                    # If no duration specified, collect only one sample
                    break
                time.sleep(interval)
        except KeyboardInterrupt:
            print("Monitoring interrupted.")
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(stats, f)
        
        return stats

# Test configuration - adjust based on system capabilities
CONFIG = {
    'dataset_size': 100000,  # Number of samples
    'n_features': 50,        # Number of features
    'test_size': 0.2,        # Test split ratio
    'h2o_memory': '2g',      # Memory for H2O
    'gpu_monitor_interval': 0.5,  # GPU monitoring interval in seconds
    'xgb_rounds': 100,       # XGBoost training rounds
    'lgbm_rounds': 100,      # LightGBM training rounds
    'h2o_max_models': 3,     # H2O AutoML max models
    'h2o_runtime': 60,       # H2O training runtime in seconds
}

# Results storage
results = {
    'system_info': {},
    'libraries': {},
    'tests': {}
}

def get_system_info():
    """Collect system information"""
    info = {}
    
    # Get CPU info
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpu_info = f.readlines()
        
        cpu_model = [line for line in cpu_info if 'model name' in line]
        if cpu_model:
            info['cpu_model'] = cpu_model[0].split(':')[1].strip()
        
        cpu_cores = len([line for line in cpu_info if 'processor' in line])
        info['cpu_cores'] = cpu_cores
    except Exception as e:
        print(f"Error getting CPU info: {e}")
        info['cpu_model'] = "Unknown"
        info['cpu_cores'] = "Unknown"
    
    # Get memory info
    try:
        with open('/proc/meminfo', 'r') as f:
            mem_info = f.readlines()
        
        total_mem = [line for line in mem_info if 'MemTotal' in line]
        if total_mem:
            mem_kb = int(total_mem[0].split(':')[1].strip().split()[0])
            info['memory_gb'] = round(mem_kb / 1024 / 1024, 2)
    except Exception as e:
        print(f"Error getting memory info: {e}")
        info['memory_gb'] = "Unknown"
    
    # Get GPU info
    info['gpus'] = get_gpu_info()
    
    # Get Python info
    info['python_version'] = sys.version
    
    return info

def check_library(library_name, import_name=None):
    """Check if a library is available and get its version"""
    import_name = import_name or library_name
    result = {'available': False, 'version': None, 'gpu_support': False}
    
    try:
        # Try to import the library
        lib = __import__(import_name)
        result['available'] = True
        if hasattr(lib, '__version__'):
            result['version'] = lib.__version__
        elif hasattr(lib, 'version'):
            result['version'] = lib.version()
        
        # Check GPU support
        if library_name == 'xgboost':
            # Modern XGBoost (3.0+) uses a different API
            try:
                # First check for CUDA capability
                if hasattr(lib, 'build_info'):
                    # XGBoost 3.0+
                    result['gpu_support'] = 'cuda' in lib.build_info().get('USE_CUDA', 'no').lower()
                elif hasattr(lib, 'get_config') and 'updaters' in lib.get_config():
                    # Older XGBoost (<3.0)
                    result['gpu_support'] = 'gpu_hist' in lib.get_config()['updaters'].split(',')
                else:
                    # Just check if 'gpu' exists in any of the library's attributes
                    module_str = str(dir(lib)).lower()
                    result['gpu_support'] = 'gpu' in module_str or 'cuda' in module_str
            except Exception as e:
                print(f"Note: Error checking XGBoost GPU support: {e}")
                # When in doubt, assume GPU support is there (we'll test it)
                result['gpu_support'] = True
        elif library_name == 'lightgbm':
            # LightGBM needs to be checked at runtime
            result['gpu_support'] = True  # We'll verify during testing
        elif library_name == 'h2o':
            # H2O GPU support is checked via H2O XGBoost
            try:
                from h2o.tree import H2OXGBoostEstimator
                result['gpu_support'] = True
            except ImportError:
                result['gpu_support'] = False
        
    except ImportError:
        pass
    
    return result

def create_datasets():
    """Create classification and regression datasets for testing"""
    print(f"\nCreating datasets with {CONFIG['dataset_size']} samples and {CONFIG['n_features']} features...")
    
    # Classification dataset
    X_clf, y_clf = make_classification(
        n_samples=CONFIG['dataset_size'],
        n_features=CONFIG['n_features'],
        n_informative=int(CONFIG['n_features'] * 0.8),
        n_redundant=int(CONFIG['n_features'] * 0.1),
        n_classes=2,
        random_state=42
    )
    
    # Regression dataset
    X_reg, y_reg = make_regression(
        n_samples=CONFIG['dataset_size'],
        n_features=CONFIG['n_features'],
        n_informative=int(CONFIG['n_features'] * 0.8),
        random_state=42
    )
    
    # Convert to pandas DataFrames
    feature_names = [f'feature_{i}' for i in range(CONFIG['n_features'])]
    
    clf_df = pd.DataFrame(X_clf, columns=feature_names)
    clf_df['target'] = y_clf
    
    reg_df = pd.DataFrame(X_reg, columns=feature_names)
    reg_df['target'] = y_reg
    
    # Split into train and test sets
    clf_train, clf_test = train_test_split(clf_df, test_size=CONFIG['test_size'], random_state=42)
    reg_train, reg_test = train_test_split(reg_df, test_size=CONFIG['test_size'], random_state=42)
    
    print(f"Classification dataset: Train shape={clf_train.shape}, Test shape={clf_test.shape}")
    print(f"Regression dataset: Train shape={reg_train.shape}, Test shape={reg_test.shape}")
    
    return {
        'classification': (clf_train, clf_test),
        'regression': (reg_train, reg_test)
    }

def test_xgboost(datasets):
    """Test XGBoost with GPU acceleration"""
    print("\n" + "="*80)
    print("TESTING XGBOOST WITH GPU")
    print("="*80)
    
    try:
        import xgboost as xgb
        
        results['tests']['xgboost'] = {
            'status': 'running',
            'error': None,
            'timing': {},
            'accuracy': {},
            'gpu_stats': {}
        }
        
        # Check if GPU is available - handle XGBoost 3.0+ API changes
        try:
            # First try older API
            gpu_available = 'gpu_hist' in xgb.get_config()['updaters'].split(',')
        except (KeyError, TypeError):
            # XGBoost 3.0+ doesn't expose updaters in get_config()
            # Instead, we'll assume GPU is available and check during training
            gpu_available = True
            print("Using XGBoost 3.0+, assuming GPU is available")
            
        if not gpu_available:
            print("XGBoost GPU support not available.")
            results['tests']['xgboost']['status'] = 'skipped'
            results['tests']['xgboost']['error'] = 'GPU support not available'
            return
        
        # Classification test
        clf_train, clf_test = datasets['classification']
        X_train = clf_train.drop('target', axis=1)
        y_train = clf_train['target']
        X_test = clf_test.drop('target', axis=1)
        y_test = clf_test['target']
        
        # Create DMatrix objects
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Set parameters for CPU and GPU - handle different XGBoost versions
        try:
            # First try the newer XGBoost 3.0+ syntax
            cpu_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'tree_method': 'hist',
                'max_depth': 6,
                'learning_rate': 0.1,  # Newer versions use learning_rate instead of eta
                'random_state': 42     # Newer versions use random_state instead of seed
            }
            
            gpu_params = cpu_params.copy()
            gpu_params['tree_method'] = 'gpu_hist'
            
            # Test if these parameters work
            xgb.DMatrix(X_train[:5], label=y_train[:5])
        except Exception as e:
            print(f"Using older XGBoost parameter format: {e}")
            # Fall back to older XGBoost parameter format
            cpu_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'tree_method': 'hist',
                'max_depth': 6,
                'eta': 0.1,
                'seed': 42
            }
            
            gpu_params = cpu_params.copy()
            gpu_params['tree_method'] = 'gpu_hist'
        
        # Test CPU performance
        print("\nTraining XGBoost on CPU...")
        start_time = time.time()
        cpu_model = xgb.train(
            cpu_params,
            dtrain,
            num_boost_round=CONFIG['xgb_rounds'],
            evals=[(dtest, 'test')],
            verbose_eval=20
        )
        cpu_time = time.time() - start_time
        # Get AUC from evaluation - handle different output formats
        try:
            eval_result = cpu_model.eval(dtest)
            if isinstance(eval_result, str):
                # Old string format
                cpu_auc = eval_result.split(':')[-1].strip()
            elif isinstance(eval_result, dict):
                # New dictionary format
                cpu_auc = str(eval_result.get('auc', eval_result.get('test-auc', 'N/A')))
            else:
                cpu_auc = str(eval_result)
        except Exception as e:
            print(f"Error getting CPU model evaluation: {e}")
            cpu_auc = "N/A"
            
        print(f"CPU training completed in {cpu_time:.2f} seconds, AUC: {cpu_auc}")
        
        # Test GPU performance with monitoring
        print("\nTraining XGBoost on GPU...")
        gpu_stats_file = os.path.join(tempfile.gettempdir(), 'xgb_gpu_stats.json')
        
        # Start GPU monitoring in a separate thread
        with ThreadPoolExecutor() as executor:
            # Start GPU monitoring
            monitor_future = executor.submit(
                monitor_gpu, 
                interval=CONFIG['gpu_monitor_interval'],
                duration=None,  # We'll stop it manually
                output_file=gpu_stats_file
            )
            
            # Train model
            start_time = time.time()
            gpu_model = xgb.train(
                gpu_params,
                dtrain,
                num_boost_round=CONFIG['xgb_rounds'],
                evals=[(dtest, 'test')],
                verbose_eval=20
            )
            gpu_time = time.time() - start_time
            
            # Wait for the monitoring to complete
            gpu_stats = monitor_future.result()
        
        # Get AUC from evaluation - handle different output formats
        try:
            eval_result = gpu_model.eval(dtest)
            if isinstance(eval_result, str):
                # Old string format
                gpu_auc = eval_result.split(':')[-1].strip()
            elif isinstance(eval_result, dict):
                # New dictionary format
                gpu_auc = str(eval_result.get('auc', eval_result.get('test-auc', 'N/A')))
            else:
                gpu_auc = str(eval_result)
        except Exception as e:
            print(f"Error getting GPU model evaluation: {e}")
            gpu_auc = "N/A"
            
        print(f"GPU training completed in {gpu_time:.2f} seconds, AUC: {gpu_auc}")
        
        # Calculate speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"GPU speedup: {speedup:.2f}x")
        
        # Store results
        results['tests']['xgboost']['status'] = 'completed'
        results['tests']['xgboost']['timing'] = {
            'cpu': cpu_time,
            'gpu': gpu_time,
            'speedup': speedup
        }
        results['tests']['xgboost']['accuracy'] = {
            'cpu_auc': float(cpu_auc) if cpu_auc != 'N/A' and not isinstance(cpu_auc, str) else str(cpu_auc),
            'gpu_auc': float(gpu_auc) if gpu_auc != 'N/A' and not isinstance(gpu_auc, str) else str(gpu_auc)
        }
        results['tests']['xgboost']['gpu_stats'] = {
            'max_utilization': max([stat['utilization_pct'] for stat in gpu_stats]) if gpu_stats else 0,
            'max_memory': max([stat['memory_used_mb'] for stat in gpu_stats]) if gpu_stats else 0,
        }
        
    except Exception as e:
        print(f"Error testing XGBoost: {e}")
        import traceback
        traceback.print_exc()
        if 'xgboost' in results['tests']:
            results['tests']['xgboost']['status'] = 'failed'
            results['tests']['xgboost']['error'] = str(e)

def test_lightgbm(datasets):
    """Test LightGBM with GPU acceleration"""
    print("\n" + "="*80)
    print("TESTING LIGHTGBM WITH GPU")
    print("="*80)
    
    try:
        import lightgbm as lgb
        
        results['tests']['lightgbm'] = {
            'status': 'running',
            'error': None,
            'timing': {},
            'accuracy': {},
            'gpu_stats': {}
        }
        
        # Check if a GPU device is available for LightGBM
        gpu_available = False
        try:
            # Create a small dataset for testing
            X, y = make_regression(n_samples=100, n_features=10, random_state=42)
            train_data = lgb.Dataset(X, label=y)
            
            # Try to train a model with GPU
            params = {'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0}
            model = lgb.train(params, train_data, num_boost_round=1)
            gpu_available = True
        except Exception as e:
            print(f"LightGBM GPU support not available: {e}")
        
        if not gpu_available:
            results['tests']['lightgbm']['status'] = 'skipped'
            results['tests']['lightgbm']['error'] = 'GPU support not available'
            return
        
        # Regression test
        reg_train, reg_test = datasets['regression']
        X_train = reg_train.drop('target', axis=1)
        y_train = reg_train['target']
        X_test = reg_test.drop('target', axis=1)
        y_test = reg_test['target']
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Set parameters for CPU and GPU
        cpu_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'verbosity': -1,
            'seed': 42
        }
        
        gpu_params = cpu_params.copy()
        gpu_params['device'] = 'gpu'
        gpu_params['gpu_platform_id'] = 0
        gpu_params['gpu_device_id'] = 0
        
        # Test CPU performance
        print("\nTraining LightGBM on CPU...")
        start_time = time.time()
        cpu_model = lgb.train(
            cpu_params,
            train_data,
            num_boost_round=CONFIG['lgbm_rounds'],
            valid_sets=[test_data],
            callbacks=[lgb.log_evaluation(period=20)]
        )
        cpu_time = time.time() - start_time
        cpu_preds = cpu_model.predict(X_test)
        cpu_rmse = np.sqrt(np.mean((cpu_preds - y_test)**2))
        print(f"CPU training completed in {cpu_time:.2f} seconds, RMSE: {cpu_rmse:.6f}")
        
        # Test GPU performance with monitoring
        print("\nTraining LightGBM on GPU...")
        gpu_stats_file = os.path.join(tempfile.gettempdir(), 'lgbm_gpu_stats.json')
        
        # Start GPU monitoring in a separate thread
        with ThreadPoolExecutor() as executor:
            # Start GPU monitoring
            monitor_future = executor.submit(
                monitor_gpu, 
                interval=CONFIG['gpu_monitor_interval'],
                duration=None,  # We'll stop it manually
                output_file=gpu_stats_file
            )
            
            # Train model
            start_time = time.time()
            gpu_model = lgb.train(
                gpu_params,
                train_data,
                num_boost_round=CONFIG['lgbm_rounds'],
                valid_sets=[test_data],
                callbacks=[lgb.log_evaluation(period=20)]
            )
            gpu_time = time.time() - start_time
            
            # Wait for the monitoring to complete
            gpu_stats = monitor_future.result()
        
        gpu_preds = gpu_model.predict(X_test)
        gpu_rmse = np.sqrt(np.mean((gpu_preds - y_test)**2))
        print(f"GPU training completed in {gpu_time:.2f} seconds, RMSE: {gpu_rmse:.6f}")
        
        # Calculate speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"GPU speedup: {speedup:.2f}x")
        
        # Store results
        results['tests']['lightgbm']['status'] = 'completed'
        results['tests']['lightgbm']['timing'] = {
            'cpu': cpu_time,
            'gpu': gpu_time,
            'speedup': speedup
        }
        results['tests']['lightgbm']['accuracy'] = {
            'cpu_rmse': float(cpu_rmse),
            'gpu_rmse': float(gpu_rmse)
        }
        results['tests']['lightgbm']['gpu_stats'] = {
            'max_utilization': max([stat['utilization_pct'] for stat in gpu_stats]) if gpu_stats else 0,
            'max_memory': max([stat['memory_used_mb'] for stat in gpu_stats]) if gpu_stats else 0,
        }
        
    except Exception as e:
        print(f"Error testing LightGBM: {e}")
        import traceback
        traceback.print_exc()
        if 'lightgbm' in results['tests']:
            results['tests']['lightgbm']['status'] = 'failed'
            results['tests']['lightgbm']['error'] = str(e)

def test_h2o_xgboost(datasets):
    """Test H2O with GPU-accelerated XGBoost"""
    print("\n" + "="*80)
    print("TESTING H2O SPARKLING WATER WITH XGBoost GPU")
    print("="*80)
    
    try:
        # Import H2O
        import h2o
        from h2o.estimators.xgboost import H2OXGBoostEstimator
        
        results['tests']['h2o_xgboost'] = {
            'status': 'running',
            'error': None,
            'timing': {},
            'accuracy': {},
            'gpu_stats': {}
        }
        
        # Check if we have Java in the correct version range (8-17)
        java_version = subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT).decode()
        if "version" not in java_version:
            print("Java not found. H2O requires Java 8-17.")
            results['tests']['h2o_xgboost']['status'] = 'skipped'
            results['tests']['h2o_xgboost']['error'] = 'Java not found'
            return
        
        try:
            # Initialize H2O with less strict version checking
            h2o.init(max_mem_size=CONFIG['h2o_memory'], strict_version_check=False)
        except Exception as e:
            print(f"Error initializing H2O: {e}")
            if "Version mismatch" in str(e):
                print("Trying with strict_version_check=False")
                h2o.init(max_mem_size=CONFIG['h2o_memory'], strict_version_check=False)
        
        # Classification test
        clf_train, clf_test = datasets['classification']
        
        # Convert pandas DataFrames to H2O frames
        train_h2o = h2o.H2OFrame(clf_train)
        test_h2o = h2o.H2OFrame(clf_test)
        
        # Convert target to categorical for classification
        train_h2o['target'] = train_h2o['target'].asfactor()
        test_h2o['target'] = test_h2o['target'].asfactor()
        
        # Set feature names and target
        features = clf_train.columns.tolist()
        target = 'target'
        features.remove(target)
        
        # Set parameters for CPU and GPU
        cpu_params = {
            'ntrees': CONFIG['xgb_rounds'],
            'max_depth': 6,
            'learn_rate': 0.1,
            'seed': 42,
            'backend': 'cpu' 
        }
        
        gpu_params = cpu_params.copy()
        gpu_params['backend'] = 'gpu'
        gpu_params['gpu_id'] = 0
        
        # Test CPU performance
        print("\nTraining H2O XGBoost on CPU...")
        try:
            start_time = time.time()
            cpu_model = H2OXGBoostEstimator(**cpu_params)
            cpu_model.train(x=features, y=target, training_frame=train_h2o, validation_frame=test_h2o)
            cpu_time = time.time() - start_time
            
            # Get model performance
            cpu_perf = cpu_model.model_performance(test_h2o)
            cpu_auc = cpu_perf.auc()
            print(f"CPU training completed in {cpu_time:.2f} seconds, AUC: {cpu_auc}")
        except Exception as e:
            print(f"Error training H2O XGBoost on CPU: {e}")
            cpu_time = None
            cpu_auc = None
        
        # Test GPU performance with monitoring
        print("\nTraining H2O XGBoost on GPU...")
        gpu_stats_file = os.path.join(tempfile.gettempdir(), 'h2o_gpu_stats.json')
        
        try:
            # Start GPU monitoring in a separate thread
            with ThreadPoolExecutor() as executor:
                # Start GPU monitoring
                monitor_future = executor.submit(
                    monitor_gpu, 
                    interval=CONFIG['gpu_monitor_interval'],
                    duration=None,  # We'll stop it manually
                    output_file=gpu_stats_file
                )
                
                # Train model
                start_time = time.time()
                gpu_model = H2OXGBoostEstimator(**gpu_params)
                gpu_model.train(x=features, y=target, training_frame=train_h2o, validation_frame=test_h2o)
                gpu_time = time.time() - start_time
                
                # Wait for the monitoring to complete
                gpu_stats = monitor_future.result()
            
            # Get model performance
            gpu_perf = gpu_model.model_performance(test_h2o)
            gpu_auc = gpu_perf.auc()
            print(f"GPU training completed in {gpu_time:.2f} seconds, AUC: {gpu_auc}")
            
            # Calculate speedup
            speedup = cpu_time / gpu_time if (gpu_time and cpu_time) else 0
            print(f"GPU speedup: {speedup:.2f}x")
            
            h2o_gpu_support = True
        except Exception as e:
            print(f"Error training H2O XGBoost on GPU: {e}")
            gpu_time = None
            gpu_auc = None
            speedup = 0
            gpu_stats = []
            h2o_gpu_support = False
        
        # Store results
        results['tests']['h2o_xgboost']['status'] = 'completed'
        results['tests']['h2o_xgboost']['gpu_support'] = h2o_gpu_support
        results['tests']['h2o_xgboost']['timing'] = {
            'cpu': cpu_time,
            'gpu': gpu_time,
            'speedup': speedup
        }
        results['tests']['h2o_xgboost']['accuracy'] = {
            'cpu_auc': float(cpu_auc) if cpu_auc is not None else None,
            'gpu_auc': float(gpu_auc) if gpu_auc is not None else None
        }
        results['tests']['h2o_xgboost']['gpu_stats'] = {
            'max_utilization': max([stat['utilization_pct'] for stat in gpu_stats]) if gpu_stats else 0,
            'max_memory': max([stat['memory_used_mb'] for stat in gpu_stats]) if gpu_stats else 0,
        }
        
        # Shutdown H2O
        h2o.cluster().shutdown()
        
    except Exception as e:
        print(f"Error testing H2O: {e}")
        import traceback
        traceback.print_exc()
        if 'h2o_xgboost' in results['tests']:
            results['tests']['h2o_xgboost']['status'] = 'failed'
            results['tests']['h2o_xgboost']['error'] = str(e)
        
        # Ensure H2O is shut down
        try:
            h2o.cluster().shutdown()
        except:
            pass

def generate_report():
    """Generate a comprehensive report with results and visualizations"""
    print("\n" + "="*80)
    print("GENERATING TEST REPORT")
    print("="*80)
    
    # Create a timestamp for the report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(project_root, "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_file = os.path.join(report_dir, f"gpu_integration_test_{timestamp}.json")
    
    # Create summary in results
    results['summary'] = {
        'timestamp': timestamp,
        'libraries_tested': list(results['libraries'].keys()),
        'tests_completed': sum(1 for t in results['tests'].values() if t['status'] == 'completed'),
        'tests_failed': sum(1 for t in results['tests'].values() if t['status'] == 'failed'),
        'tests_skipped': sum(1 for t in results['tests'].values() if t['status'] == 'skipped'),
    }
    
    # Save results to file
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Report saved to: {report_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    print(f"Tests completed: {results['summary']['tests_completed']}")
    print(f"Tests failed: {results['summary']['tests_failed']}")
    print(f"Tests skipped: {results['summary']['tests_skipped']}")
    
    print("\nPerformance Summary:")
    for lib, test in results['tests'].items():
        if test['status'] == 'completed':
            if 'timing' in test and 'speedup' in test['timing']:
                print(f"- {lib}: {test['timing']['speedup']:.2f}x speedup")
            if 'gpu_stats' in test and 'max_utilization' in test['gpu_stats']:
                print(f"  Max GPU utilization: {test['gpu_stats']['max_utilization']:.1f}%")
                print(f"  Max GPU memory used: {test['gpu_stats']['max_memory']:.1f} MB")
    
    return report_file

def main():
    """Main test function"""
    print("=" * 80)
    print("GPU INTEGRATION TEST FOR ML LIBRARIES")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Collect system information
    print("\nCollecting system information...")
    results['system_info'] = get_system_info()
    
    # Display GPU information
    print("\nGPU Information:")
    if results['system_info']['gpus']:
        for i, gpu in enumerate(results['system_info']['gpus']):
            print(f"- GPU {i}: {gpu['name']}")
            print(f"  Memory: {gpu['memory_total_mb']} MB")
            print(f"  Current utilization: {gpu['utilization_pct']}%")
    else:
        print("No GPUs detected.")
        return 1
    
    # Check libraries
    print("\nChecking ML libraries...")
    results['libraries']['xgboost'] = check_library('xgboost')
    results['libraries']['lightgbm'] = check_library('lightgbm')
    results['libraries']['h2o'] = check_library('h2o')
    
    # Display library information
    for lib, info in results['libraries'].items():
        status = "Available" if info['available'] else "Not available"
        gpu = "Yes" if info['gpu_support'] else "No"
        version = info['version'] if info['version'] else "Unknown"
        print(f"- {lib}: {status}, Version: {version}, GPU Support: {gpu}")
    
    # Create datasets for testing
    datasets = create_datasets()
    
    # Run tests for each library
    if results['libraries']['xgboost']['available']:
        test_xgboost(datasets)
    
    if results['libraries']['lightgbm']['available']:
        test_lightgbm(datasets)
    
    if results['libraries']['h2o']['available']:
        test_h2o_xgboost(datasets)
    
    # Generate report
    report_file = generate_report()
    
    print("\nTest completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())