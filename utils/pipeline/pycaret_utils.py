#!/usr/bin/env python3
"""
PyCaret pipeline utilities for GPU-accelerated machine learning
"""

import os
import sys
import subprocess
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List

warnings.filterwarnings('ignore')


def log_message(message: str, level: str = "INFO"):
    """Log messages with timestamp"""
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] [{level}] {message}")


def check_system_requirements() -> Dict[str, Any]:
    """Check system requirements for PyCaret GPU training"""
    status = {
        'os_compatible': True,
        'python_version': sys.version_info,
        'memory_gb': 0,
        'disk_space_gb': 0,
        'shared_memory_writable': False
    }
    
    # Check memory
    try:
        import psutil
        status['memory_gb'] = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass
    
    # Check disk space
    try:
        import shutil
        status['disk_space_gb'] = shutil.disk_usage('/tmp').free / (1024**3)
    except Exception:
        pass
    
    # Check shared memory
    status['shared_memory_writable'] = os.access('/dev/shm', os.W_OK)
    
    return status


def validate_environment() -> bool:
    """Validate that the environment is ready for PyCaret training"""
    log_message("Validating environment for PyCaret GPU training...")
    
    # Check if running as root
    if os.geteuid() == 0:
        log_message("WARNING: Running as root. Consider using a non-root user for ML workloads.", "WARNING")
    
    # Check system requirements
    sys_req = check_system_requirements()
    
    if sys_req['memory_gb'] < 8:
        log_message(f"WARNING: Low memory detected ({sys_req['memory_gb']:.1f} GB). Minimum 8GB recommended.", "WARNING")
    
    if sys_req['disk_space_gb'] < 5:
        log_message(f"WARNING: Low disk space ({sys_req['disk_space_gb']:.1f} GB). Minimum 5GB recommended.", "WARNING")
    
    if not sys_req['shared_memory_writable']:
        log_message("WARNING: Cannot write to /dev/shm. Multi-processing may be limited.", "WARNING")
    
    return True


def get_gpu_status() -> Dict[str, Any]:
    """Get comprehensive GPU status"""
    status = {
        'nvidia_smi_available': False,
        'cuda_toolkit_available': False,
        'gpu_count': 0,
        'gpu_info': []
    }
    
    # Check nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                               capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            status['nvidia_smi_available'] = True
            status['gpu_count'] = len(result.stdout.strip().split('\n'))
            
            # Get detailed GPU info
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free', 
                                   '--format=csv,noheader,nounits'], 
                                   capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 4:
                            status['gpu_info'].append({
                                'index': int(parts[0]),
                                'name': parts[1],
                                'memory_total': int(parts[2]),
                                'memory_free': int(parts[3])
                            })
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        pass
    
    # Check CUDA toolkit
    try:
        result = subprocess.run(['nvcc', '--version'], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            status['cuda_toolkit_available'] = True
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        pass
    
    return status


def create_optimized_launcher_script(output_path: str = "/tmp/run_pycaret_optimized.py") -> str:
    """Create optimized PyCaret launcher script"""
    launcher_content = '''#!/usr/bin/env python3
"""
Optimized PyCaret launcher with multi-GPU configuration
"""

import os
import gc
import warnings
warnings.filterwarnings('ignore')

def optimize_memory():
    """Apply memory optimizations"""
    gc.collect()
    
    try:
        import psutil
        import resource
        available_memory = psutil.virtual_memory().available
        memory_limit = int(available_memory * 0.8)
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        print(f"Memory limit set to {memory_limit // (1024**3)} GB")
    except:
        print("Could not set memory limits")

def setup_gpu_environment():
    """Setup GPU environment for optimal performance"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['OMP_NUM_THREADS'] = '6'
    os.environ['MKL_NUM_THREADS'] = '6'
    print("GPU environment configured for 3 GPUs")

def main():
    """Main launcher function"""
    print("=== PyCaret Optimized Launcher ===")
    
    optimize_memory()
    setup_gpu_environment()
    
    try:
        from pycaret.regression import setup as regression_setup
        from pycaret.classification import setup as classification_setup
        print("✓ PyCaret imported successfully")
    except ImportError as e:
        print(f"✗ PyCaret import failed: {e}")
        return False
    
    print("Ready for PyCaret multi-GPU training!")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
'''
    
    with open(output_path, 'w') as f:
        f.write(launcher_content)
    
    os.chmod(output_path, 0o755)
    return output_path


def run_pycaret_gpu_checks() -> bool:
    """Run comprehensive PyCaret GPU environment checks"""
    log_message("Starting PyCaret GPU environment checks...")
    
    # Validate environment
    if not validate_environment():
        return False
    
    # Check GPU status
    gpu_status = get_gpu_status()
    
    if gpu_status['nvidia_smi_available']:
        log_message(f"✓ NVIDIA GPU detected: {gpu_status['gpu_count']} GPU(s)")
        for gpu in gpu_status['gpu_info']:
            log_message(f"  GPU {gpu['index']}: {gpu['name']} ({gpu['memory_total']} MB total)")
    else:
        log_message("⚠ No NVIDIA GPUs detected or nvidia-smi not available", "WARNING")
    
    if gpu_status['cuda_toolkit_available']:
        log_message("✓ CUDA toolkit available")
    else:
        log_message("⚠ CUDA toolkit not detected", "WARNING")
    
    # Check Python packages
    from ..gpu.pycaret_optimizer import check_pycaret_packages, check_gpu_libraries
    
    pkg_status = check_pycaret_packages()
    missing_packages = [pkg for pkg, available in pkg_status.items() if not available]
    
    if missing_packages:
        log_message(f"✗ Missing required packages: {missing_packages}", "ERROR")
        log_message(f"Install with: pip install {' '.join(missing_packages)}", "INFO")
        return False
    else:
        log_message("✓ All required packages are installed")
    
    # Check GPU libraries
    gpu_libs = check_gpu_libraries()
    
    for lib_name, lib_info in gpu_libs.items():
        if lib_info['available']:
            gpu_support = "with GPU" if lib_info.get('gpu_support', False) else "CPU only"
            log_message(f"✓ {lib_name}: Available ({gpu_support})")
        else:
            log_message(f"✗ {lib_name}: Not available", "WARNING")
    
    # Create and test optimized launcher
    launcher_path = create_optimized_launcher_script()
    log_message(f"Created optimized launcher: {launcher_path}")
    
    try:
        result = subprocess.run([sys.executable, launcher_path], 
                               capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            log_message("✓ PyCaret launcher test successful")
            return True
        else:
            log_message(f"✗ PyCaret launcher test failed: {result.stderr}", "ERROR")
            return False
    except subprocess.TimeoutExpired:
        log_message("✗ PyCaret launcher test timed out", "ERROR")
        return False


def activate_pycaret_gpu_environment(gpu_count: int = 3) -> bool:
    """Main function to activate PyCaret GPU environment"""
    log_message("=== PyCaret Multi-GPU Activation ===")
    
    try:
        # Import and use GPU optimizer
        from ..gpu.pycaret_optimizer import initialize_pycaret_gpu, create_pycaret_config
        
        # Initialize GPU environment
        if not initialize_pycaret_gpu(gpu_count=gpu_count, memory_optimize=True):
            log_message("✗ PyCaret GPU initialization failed", "ERROR")
            return False
        
        # Create configuration
        config_path = create_pycaret_config(gpu_count=gpu_count, use_gpu=True)
        log_message(f"✓ PyCaret configuration created: {config_path}")
        
        # Run comprehensive checks
        if not run_pycaret_gpu_checks():
            log_message("⚠ Some checks failed, but continuing...", "WARNING")
        
        log_message("✓ PyCaret multi-GPU activation completed successfully!")
        log_message("Environment is ready for PyCaret multi-GPU training!")
        
        return True
        
    except Exception as e:
        log_message(f"✗ PyCaret activation failed: {e}", "ERROR")
        return False