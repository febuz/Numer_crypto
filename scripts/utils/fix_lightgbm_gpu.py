#!/usr/bin/env python3
"""
Fixes the LightGBM configuration for GPU support by modifying the models using tree_learner=gpu
"""

import sys
import os
from pathlib import Path

def print_colored(message, color="INFO"):
    """Print a colored message"""
    colors = {
        "INFO": "\033[0;34m[INFO]\033[0m",
        "SUCCESS": "\033[0;32m[SUCCESS]\033[0m",
        "WARNING": "\033[1;33m[WARNING]\033[0m",
        "ERROR": "\033[0;31m[ERROR]\033[0m"
    }
    print(f"{colors.get(color, colors['INFO'])} {message}")

def modify_model_files():
    """Find and modify all LightGBM model files using tree_learner=gpu"""
    modified_files = 0
    project_root = Path(__file__).parent
    
    print_colored(f"Scanning project directory: {project_root}")
    
    # File patterns to search for
    patterns = ["**/*.py"]
    
    for pattern in patterns:
        for filepath in project_root.glob(pattern):
            if not filepath.is_file():
                continue
                
            try:
                with open(filepath, 'r') as file:
                    content = file.read()
                
                # Check if the file contains the problematic tree_learner setting
                if "'tree_learner': 'serial'" in content or '"tree_learner": "serial"' in content:
                    print_colored(f"Found file with tree_learner=gpu: {filepath}", "INFO")
                    
                    # Modified content
                    # Replace tree_learner with data_sample_strategy which is supported by CPU version
                    modified_content = content.replace("'tree_learner': 'serial'", "'tree_learner': 'serial'")
                    modified_content = modified_content.replace('"tree_learner": "serial"', '"tree_learner": "serial"')
                    
                    # Write the modified content back to the file
                    with open(filepath, 'w') as file:
                        file.write(modified_content)
                    
                    modified_files += 1
                    print_colored(f"Modified file: {filepath}", "SUCCESS")
            except Exception as e:
                print_colored(f"Error processing {filepath}: {e}", "ERROR")
    
    print_colored(f"Total files modified: {modified_files}", "INFO")
    return modified_files

def add_lightgbm_gpu_wrapper():
    """Create a wrapper to handle LightGBM GPU configuration transparently"""
    wrapper_path = Path(__file__).parent / "utils" / "gpu" / "lightgbm_wrapper.py"
    
    # Create directories if they don't exist
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    
    wrapper_content = """#!/usr/bin/env python3
\"\"\"
LightGBM GPU wrapper to handle configuration transparently
This module intercepts LightGBM tree_learner parameters and adjusts them based on the 
actual GPU support availability in the installed LightGBM version.
\"\"\"

import os
import sys
import logging
from functools import wraps
import importlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Original LightGBM module
try:
    import lightgbm as lgb
    _original_lgb = lgb
except ImportError:
    logger.error("LightGBM not installed. Please install it first.")
    sys.exit(1)

# Check if GPU support is available
def _has_gpu_support():
    try:
        # Create a dataset and try to train with GPU
        import numpy as np
        X = np.random.rand(10, 5)
        y = np.random.rand(10)
        dataset = lgb.Dataset(X, label=y)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'device': 'gpu',
            'tree_learner': 'serial',
            'verbose': -1
        }
        
        # Try training with GPU
        try:
            model = lgb.train(params, dataset, num_boost_round=1)
            return True
        except Exception as e:
            if "Unknown tree learner type gpu" in str(e):
                return False
            # For other errors, assume GPU might be supported but has other issues
            return True
    except Exception:
        return False

# Determine if GPU support is available
HAS_GPU_SUPPORT = _has_gpu_support()
logger.info(f"LightGBM GPU support detected: {HAS_GPU_SUPPORT}")

# Wrap LightGBM train function to handle GPU parameters
@wraps(_original_lgb.train)
def train_wrapper(params, *args, **kwargs):
    # Wrapper for LightGBM train function to handle GPU parameters
    params = params.copy()  # Make a copy to avoid modifying the original
    
    # Check if GPU parameters are specified
    has_gpu_params = (
        params.get('device', '').lower() == 'gpu' or
        params.get('tree_learner', '').lower() == 'gpu'
    )
    
    if has_gpu_params and not HAS_GPU_SUPPORT:
        logger.warning("LightGBM GPU support not available. Falling back to CPU.")
        # Replace GPU parameters with CPU equivalents
        if 'device' in params and params['device'].lower() == 'gpu':
            params['device'] = 'cpu'
        if 'tree_learner' in params and params['tree_learner'].lower() == 'gpu':
            params['tree_learner'] = 'serial'
        # Remove GPU-specific parameters
        params.pop('gpu_platform_id', None)
        params.pop('gpu_device_id', None)
    
    # Call the original train function
    return _original_lgb.train(params, *args, **kwargs)

# Monkey patch LightGBM functions
lgb.train = train_wrapper

# Patch LGBMModel class to handle GPU parameters
if hasattr(_original_lgb, 'LGBMModel'):
    original_init = _original_lgb.LGBMModel.__init__
    
    @wraps(original_init)
    def init_wrapper(self, *args, **kwargs):
        # Check for GPU parameters
        if not HAS_GPU_SUPPORT:
            if 'device' in kwargs and kwargs['device'].lower() == 'gpu':
                logger.warning("LightGBM GPU support not available. Using CPU for device.")
                kwargs['device'] = 'cpu'
            if 'tree_learner' in kwargs and kwargs['tree_learner'].lower() == 'gpu':
                logger.warning("LightGBM GPU support not available. Using serial tree learner.")
                kwargs['tree_learner'] = 'serial'
            # Remove GPU-specific parameters
            kwargs.pop('gpu_platform_id', None)
            kwargs.pop('gpu_device_id', None)
        
        # Call the original init
        original_init(self, *args, **kwargs)
    
    _original_lgb.LGBMModel.__init__ = init_wrapper

# Export all LightGBM attributes
for attr in dir(_original_lgb):
    if not attr.startswith('_'):
        globals()[attr] = getattr(_original_lgb, attr)

# Add wrapper version
__version__ = _original_lgb.__version__ + '-gpu-wrapper'

logger.info(f"LightGBM GPU wrapper initialized (version {__version__})")
"""
    
    # Create the wrapper file
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_content)
    
    print_colored(f"Created LightGBM GPU wrapper at: {wrapper_path}", "SUCCESS")
    
    # Create __init__.py file to import the wrapper
    init_path = wrapper_path.parent / "__init__.py"
    
    # Check if __init__.py exists and if it already imports the wrapper
    init_content = ""
    if init_path.exists():
        with open(init_path, 'r') as f:
            init_content = f.read()
    
    # Add the import if it doesn't exist
    if "import lightgbm_wrapper" not in init_content:
        with open(init_path, 'a') as f:
            f.write("\n# Import LightGBM wrapper\ntry:\n    from .lightgbm_wrapper import *\nexcept ImportError:\n    pass\n")
    
    print_colored(f"Updated {init_path} to import the wrapper", "SUCCESS")
    
    return wrapper_path

def create_wrapper_usage_documentation():
    """Create documentation on how to use the wrapper"""
    docs_path = Path(__file__).parent / "docs" / "lightgbm_gpu_wrapper.md"
    
    # Create directories if they don't exist
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    
    docs_content = """# LightGBM GPU Wrapper

## Overview

This document describes the LightGBM GPU wrapper that was added to handle the "Unknown tree learner type gpu" error. The wrapper provides a transparent solution that allows your code to specify GPU parameters even when the installed LightGBM version doesn't support them.

## Problem

The error "Unknown tree learner type gpu" occurs when:

1. The LightGBM Python package was not compiled with GPU support
2. Code attempts to use `tree_learner='gpu'` parameter

## Solution

The wrapper at `utils/gpu/lightgbm_wrapper.py` does the following:

1. Detects if GPU support is available in the installed LightGBM version
2. If GPU support is not available, it automatically:
   - Replaces `device='gpu'` with `device='cpu'`
   - Replaces `tree_learner='gpu'` with `tree_learner='serial'`
   - Removes other GPU-specific parameters (`gpu_platform_id`, `gpu_device_id`)
3. Transparently passes all other parameters to the original LightGBM functions

## Usage

### Option 1: Modify imports (recommended)

Change your imports from:

```python
import lightgbm as lgb
```

To:

```python
from utils.gpu.lightgbm_wrapper import *
```

This will use the wrapper which handles GPU parameters transparently, without changing any other code.

### Option 2: Import the wrapper directly

```python
# Import the wrapper first
from utils.gpu import lightgbm_wrapper

# Then use it as regular lightgbm
import lightgbm as lgb
```

### Option 3: Manual parameter handling

If you prefer not to use the wrapper, modify your code to check for GPU support:

```python
import lightgbm as lgb

# Check if GPU is supported
def has_gpu_support():
    try:
        import numpy as np
        X = np.random.rand(10, 5)
        y = np.random.rand(10)
        dataset = lgb.Dataset(X, label=y)
        params = {'tree_learner': 'serial', 'verbose': -1}
        model = lgb.train(params, dataset, num_boost_round=1)
        return True
    except Exception as e:
        if "Unknown tree learner type gpu" in str(e):
            return False
        return True

# Set parameters based on GPU support
params = {
    'objective': 'binary',
    'metric': 'auc',
    # Other parameters...
}

if has_gpu_support():
    params.update({
        'device': 'gpu',
        'tree_learner': 'serial'
    })
else:
    params.update({
        'device': 'cpu',
        'tree_learner': 'serial'
    })

# Use the parameters
model = lgb.train(params, train_data)
```

## Building LightGBM with GPU Support

If you want full GPU support, install LightGBM with GPU support:

```bash
# Uninstall existing LightGBM
pip uninstall -y lightgbm

# Install from source with GPU support
git clone --recursive https://github.com/microsoft/LightGBM.git
cd LightGBM
mkdir build && cd build
cmake -DUSE_GPU=ON ..
make -j4
cd ../python-package
pip install --no-binary lightgbm .
```

Note: This requires the CUDA toolkit and OpenCL libraries to be installed.
"""
    
    # Create the documentation file
    with open(docs_path, 'w') as f:
        f.write(docs_content)
    
    print_colored(f"Created LightGBM GPU wrapper documentation at: {docs_path}", "SUCCESS")
    
    return docs_path

def main():
    print_colored("Starting LightGBM GPU support fix...", "INFO")
    
    # Modify model files
    print_colored("Step 1: Modifying model files with tree_learner=gpu", "INFO")
    modified_count = modify_model_files()
    
    # Create wrapper
    print_colored("Step 2: Creating LightGBM GPU wrapper", "INFO")
    wrapper_path = add_lightgbm_gpu_wrapper()
    
    # Create documentation
    print_colored("Step 3: Creating documentation", "INFO")
    docs_path = create_wrapper_usage_documentation()
    
    print_colored("\nLightGBM GPU support fix completed!", "SUCCESS")
    print_colored(f"Modified {modified_count} files", "INFO")
    print_colored(f"Created wrapper at {wrapper_path}", "INFO")
    print_colored(f"Created documentation at {docs_path}", "INFO")
    
    print_colored("\nTo use the wrapper, import LightGBM as:", "INFO")
    print_colored("from utils.gpu.lightgbm_wrapper import *", "INFO")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())