#!/usr/bin/env python3
"""
Multi-GPU Test for PyTorch

This script demonstrates how to use multiple GPUs in parallel with PyTorch for
distributed machine learning. It creates synthetic data and trains multiple
models simultaneously on different GPUs.

Requirements:
- Multiple NVIDIA GPUs
- CUDA support
- PyTorch with CUDA support
"""

import os
import sys
import time
import json
import threading
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from threading import Event
from concurrent.futures import ThreadPoolExecutor, as_completed

# Check for required dependencies before proceeding
missing_modules = []
required_modules = ["numpy", "pandas", "matplotlib", "torch", "sklearn"]

for module in required_modules:
    try:
        __import__(module)
    except ImportError:
        missing_modules.append(module)

# If modules are missing, show a helpful error message
if missing_modules:
    print("\nERROR: The following required modules are missing:")
    for module in missing_modules:
        print(f"  - {module}")
    print("\nPlease install them using:")
    print("  pip install " + " ".join(missing_modules))
    sys.exit(1)

# Now that we've checked dependencies, import them
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# GPU monitoring utilities
def get_gpu_info():
    """Get detailed GPU information for all available GPUs"""
    try:
        # Run nvidia-smi command with more detailed information
        nvidia_smi_output = subprocess.check_output(
            ['nvidia-smi', 
             '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        
        gpu_info = []
        for line in nvidia_smi_output.strip().split('\n'):
            values = line.split(', ')
            if len(values) >= 6:
                gpu_info.append({
                    'index': int(values[0]),
                    'name': values[1],
                    'utilization_pct': float(values[2]),
                    'memory_used_mb': float(values[3]),
                    'memory_total_mb': float(values[4]),
                    'temperature_c': float(values[5])
                })
        return gpu_info
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return []

def monitor_all_gpus(stop_event, results_dict, interval=0.1):
    """Monitor all GPUs in separate thread"""
    while not stop_event.is_set():
        gpu_info = get_gpu_info()
        timestamp = time.time()
        
        for gpu in gpu_info:
            gpu_idx = gpu['index']
            # Initialize list for this GPU if it doesn't exist
            if gpu_idx not in results_dict:
                results_dict[gpu_idx] = []
                
            results_dict[gpu_idx].append({
                'timestamp': timestamp,
                'gpu_index': gpu_idx,
                'name': gpu['name'],
                'utilization': gpu['utilization_pct'],
                'memory_used': gpu['memory_used_mb'],
                'memory_total': gpu['memory_total_mb'],
                'temperature': gpu['temperature_c']
            })
        time.sleep(interval)

def create_synthetic_dataset(n_rows=100000, n_features=20, random_seed=42):
    """Create synthetic dataset for model training"""
    print(f"Creating dataset with {n_rows} samples and {n_features} features")
    X, y = make_classification(
        n_samples=n_rows,
        n_features=n_features,
        n_informative=int(n_features * 0.8),
        n_redundant=int(n_features * 0.1),
        n_classes=2,
        random_state=random_seed
    )
    
    # Convert to pandas DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df

class NeuralNetwork(nn.Module):
    """Simple neural network for binary classification"""
    def __init__(self, input_size, hidden_size=128):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer3 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.dropout(self.relu(self.layer1(x)))
        x = self.dropout(self.relu(self.layer2(x)))
        x = self.sigmoid(self.layer3(x))
        return x

def train_pytorch_on_gpu(gpu_id, data_df, model_name=None, batch_size=128, epochs=20):
    """Train PyTorch model on specific GPU"""
    model_name = model_name or f"PyTorch_Model_GPU{gpu_id}"
    print(f"\n=== Training {model_name} on GPU {gpu_id} ===")
    
    try:
        # Set specific GPU as visible
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        
        # Split data
        feature_cols = [col for col in data_df.columns if col != 'target']
        X = data_df[feature_cols].values
        y = data_df['target'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Convert to torch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Initialize model
        input_size = X_train.shape[1]
        model = NeuralNetwork(input_size=input_size)
        model.to(device)
        
        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train the model
        start_time = time.time()
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
            
            if epoch % 5 == 0 or epoch == epochs - 1:
                epoch_loss = running_loss / len(train_loader.dataset)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        training_time = time.time() - start_time
        
        # Evaluate the model
        model.eval()
        y_pred = []
        
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                y_pred.extend(outputs.cpu().numpy())
        
        y_pred = np.array(y_pred).flatten()
        auc = roc_auc_score(y_test, y_pred)
        
        print(f"GPU {gpu_id} Results:")
        print(f"  - Training Time: {training_time:.2f} seconds")
        print(f"  - AUC: {auc:.4f}")
        
        # Save model
        model_dir = Path('./models/pytorch')
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{model_name}_gpu{gpu_id}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        return {
            'gpu_id': gpu_id,
            'model_name': model_name,
            'success': True,
            'training_time': training_time,
            'auc': float(auc),
            'model_path': str(model_path)
        }
            
    except Exception as e:
        print(f"Error training on GPU {gpu_id}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'gpu_id': gpu_id,
            'model_name': model_name,
            'success': False,
            'error': str(e)
        }

def plot_multi_gpu_results(results, gpu_metrics, output_dir='./reports'):
    """Generate visualizations for multi-GPU test results"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Training times comparison
    successful_results = [r for r in results if r.get('success', False)]
    if not successful_results:
        print("No successful training results to plot")
        return
    
    # Plot training times
    plt.figure(figsize=(10, 6))
    gpu_ids = [r['gpu_id'] for r in successful_results]
    times = [r['training_time'] for r in successful_results]
    aucs = [r['auc'] for r in successful_results]
    
    bars = plt.bar(gpu_ids, times, color=['blue', 'green', 'red', 'purple'])
    plt.title('Training Time by GPU')
    plt.xlabel('GPU ID')
    plt.ylabel('Training Time (seconds)')
    
    # Add values on bars
    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        plt.annotate(f'{height:.2f}s\nAUC: {auc:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"{output_dir}/multi_gpu_pytorch_training_times_{timestamp}.png")
    
    # 2. GPU utilization over time
    # Plot each GPU's utilization on the same chart with different colors
    plt.figure(figsize=(14, 8))
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray']
    
    for idx, (gpu_id, metrics) in enumerate(gpu_metrics.items()):
        if not metrics:
            continue
            
        df = pd.DataFrame(metrics)
        start_time = df['timestamp'].min()
        df['seconds'] = df['timestamp'] - start_time
        
        color = colors[idx % len(colors)]
        plt.plot(df['seconds'], df['utilization'], 
                 label=f'GPU {gpu_id} ({df["name"].iloc[0]})', 
                 color=color)
    
    plt.title('GPU Utilization Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('GPU Utilization (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/multi_gpu_pytorch_utilization_{timestamp}.png")
    
    # 3. Memory usage over time
    plt.figure(figsize=(14, 8))
    
    for idx, (gpu_id, metrics) in enumerate(gpu_metrics.items()):
        if not metrics:
            continue
            
        df = pd.DataFrame(metrics)
        start_time = df['timestamp'].min()
        df['seconds'] = df['timestamp'] - start_time
        
        color = colors[idx % len(colors)]
        plt.plot(df['seconds'], df['memory_used'], 
                 label=f'GPU {gpu_id} ({df["name"].iloc[0]})', 
                 color=color)
    
    plt.title('GPU Memory Usage Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Memory Used (MB)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/multi_gpu_pytorch_memory_{timestamp}.png")
    
    print(f"Plots saved to {output_dir}/")

def main():
    """Main function to run multi-GPU tests with PyTorch"""
    parser = argparse.ArgumentParser(description='Multi-GPU Test for PyTorch')
    parser.add_argument('--rows', type=int, default=100000, help='Number of rows in the dataset')
    parser.add_argument('--cols', type=int, default=20, help='Number of features in the dataset')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--output-dir', type=str, default='./reports', help='Output directory for results')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MULTI-GPU TEST FOR PYTORCH")
    print("=" * 80)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return 1
    
    # Get available GPUs
    gpus = get_gpu_info()
    if not gpus:
        print("No GPUs detected. Exiting.")
        return 1
    
    num_gpus = torch.cuda.device_count()
    print(f"PyTorch can see {num_gpus} GPUs")
    
    print(f"Detected {len(gpus)} GPUs:")
    for gpu in gpus:
        print(f"  GPU {gpu['index']}: {gpu['name']}")
        print(f"    Memory: {gpu['memory_total_mb']} MB")
        print(f"    Current utilization: {gpu['utilization_pct']}%")
    
    # Create synthetic dataset (shared across all GPU tests)
    data_df = create_synthetic_dataset(args.rows, args.cols)
    
    # Start GPU monitoring in a separate thread
    gpu_metrics = {}  # Dictionary to store metrics for each GPU
    monitor_stop_event = Event()
    monitor_thread = threading.Thread(
        target=monitor_all_gpus, 
        args=(monitor_stop_event, gpu_metrics)
    )
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Train on all available GPUs in parallel
    print("\n" + "="*80)
    print(f"TRAINING PYTORCH MODELS ON {len(gpus)} GPUs")
    print("="*80)
    
    results = []
    
    # Configure parallel execution - limit based on system resources
    max_parallel = min(len(gpus), 3)  # Limit to 3 GPUs at once to avoid overloading
    
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        # Submit a task for each GPU
        futures = {}
        for i, gpu in enumerate(gpus):
            gpu_id = gpu['index']
            model_name = f"PyTorch_Model_GPU{gpu_id}"
            
            future = executor.submit(
                train_pytorch_on_gpu, 
                gpu_id, 
                data_df, 
                model_name,
                args.batch_size,
                args.epochs
            )
            futures[future] = gpu_id
        
        # Collect results as they complete
        for future in as_completed(futures):
            gpu_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Completed training on GPU {gpu_id}")
            except Exception as e:
                print(f"Error training on GPU {gpu_id}: {e}")
                results.append({
                    'gpu_id': gpu_id,
                    'success': False,
                    'error': str(e)
                })
    
    # Stop monitoring
    monitor_stop_event.set()
    monitor_thread.join()
    
    # Print summary results
    print("\n" + "="*80)
    print("MULTI-GPU TEST RESULTS")
    print("="*80)
    
    successful_count = sum(1 for r in results if r.get('success', False))
    failed_count = len(results) - successful_count
    
    print(f"Successful models: {successful_count}")
    print(f"Failed models: {failed_count}")
    
    # Print results for each GPU
    for result in results:
        gpu_id = result['gpu_id']
        if result.get('success', False):
            print(f"\nGPU {gpu_id}:")
            print(f"  Model: {result['model_name']}")
            print(f"  Training time: {result['training_time']:.2f} seconds")
            print(f"  AUC: {result['auc']:.4f}")
        else:
            print(f"\nGPU {gpu_id}: Failed - {result.get('error', 'Unknown error')}")
    
    # Print peak utilization for each GPU
    print("\nPeak GPU Utilization:")
    for gpu_id, metrics in gpu_metrics.items():
        if not metrics:
            continue
            
        df = pd.DataFrame(metrics)
        peak_util = df['utilization'].max()
        peak_memory = df['memory_used'].max()
        avg_util = df['utilization'].mean()
        
        print(f"  GPU {gpu_id}:")
        print(f"    Peak: {peak_util:.2f}%")
        print(f"    Average: {avg_util:.2f}%")
        print(f"    Peak Memory: {peak_memory:.2f} MB")
    
    # Generate plots
    plot_multi_gpu_results(results, gpu_metrics, args.output_dir)
    
    # Save detailed results to file
    timestamp = int(time.time())
    result_file = os.path.join(args.output_dir, f"multi_gpu_pytorch_{timestamp}.json")
    
    # Prepare metrics for saving (without raw monitoring data)
    metrics_summary = {}
    for gpu_id, metrics in gpu_metrics.items():
        if not metrics:
            continue
            
        df = pd.DataFrame(metrics)
        metrics_summary[str(gpu_id)] = {
            'gpu_name': df['name'].iloc[0],
            'samples_count': len(df),
            'peak_utilization': float(df['utilization'].max()),
            'avg_utilization': float(df['utilization'].mean()),
            'peak_memory': float(df['memory_used'].max()),
            'avg_memory': float(df['memory_used'].mean()),
            'peak_temperature': float(df['temperature'].max())
        }
    
    with open(result_file, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'gpu_count': len(gpus),
            'pytorch_visible_devices': num_gpus,
            'results': results,
            'metrics_summary': metrics_summary,
            'test_parameters': vars(args)
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {result_file}")
    
    # Success if at least one model trained successfully
    return 0 if successful_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main())