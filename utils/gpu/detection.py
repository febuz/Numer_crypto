#!/usr/bin/env python3
"""
GPU Detection Utilities
"""

import os
import logging
import subprocess
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


def get_available_gpus() -> List[int]:
    """Get list of available GPU IDs"""
    try:
        import torch
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
    except ImportError:
        pass
    
    try:
        # Try nvidia-smi
        result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                               capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            return list(range(len(lines)))
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        pass
    
    return []


def select_best_gpu() -> Optional[int]:
    """Select the best available GPU based on memory and utilization"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count > 0:
                # For simplicity, return GPU 0
                # In a more sophisticated implementation, we'd check memory usage
                return 0
    except ImportError:
        pass
    
    # Try using nvidia-smi to find best GPU
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu', 
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            best_gpu = 0
            best_score = -1
            
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 3:
                    gpu_id = int(parts[0])
                    memory_free = int(parts[1])
                    utilization = int(parts[2])
                    
                    # Score based on free memory and low utilization
                    score = memory_free * (100 - utilization)
                    if score > best_score:
                        best_score = score
                        best_gpu = gpu_id
            
            return best_gpu
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError, ValueError):
        pass
    
    return None


def get_gpu_info() -> Dict[str, Any]:
    """Get detailed GPU information"""
    info = {
        'available': False,
        'count': 0,
        'devices': []
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            info['available'] = True
            info['count'] = torch.cuda.device_count()
            
            for i in range(info['count']):
                device_info = {
                    'id': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_total': torch.cuda.get_device_properties(i).total_memory,
                    'memory_allocated': torch.cuda.memory_allocated(i),
                    'memory_reserved': torch.cuda.memory_reserved(i)
                }
                info['devices'].append(device_info)
    except ImportError:
        pass
    
    return info


def check_gpu_compatibility() -> bool:
    """Check if GPU setup is compatible with our requirements"""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        
        # Test basic CUDA operations
        x = torch.ones(2, 2, device='cuda')
        y = x + x
        result = y.cpu().numpy()
        
        return True
    except Exception:
        return False