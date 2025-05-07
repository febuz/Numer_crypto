#!/usr/bin/env python3
"""
Threading and parallel processing utilities.

This module provides functions for parallelizing tasks across multiple cores,
managing thread pools, and optimizing CPU usage.
"""
import os
import sys
import logging
import time
import math
import concurrent.futures
from typing import List, Dict, Tuple, Union, Optional, Callable, Any, Iterable
from functools import partial

# Add parent directory to path if needed
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.log_utils import setup_logging
from config.settings import HARDWARE_CONFIG

# Set up logging
logger = setup_logging(name=__name__, level=logging.INFO)

# Determine CPU count
CPU_COUNT = os.cpu_count() or 4
MAX_WORKERS = min(96, CPU_COUNT * 2)  # Use 2 workers per CPU, but cap at 96

def get_optimal_thread_count(memory_intensive: bool = False) -> int:
    """
    Calculate the optimal number of threads/workers based on hardware.
    
    Args:
        memory_intensive (bool): Whether the task is memory-intensive
        
    Returns:
        int: Optimal number of threads/workers
    """
    if memory_intensive:
        # For memory-intensive tasks, limit the number of threads
        # Heuristic: 1 thread per 32GB of RAM
        try:
            memory_str = HARDWARE_CONFIG.get('total_memory', '64g')
            if memory_str.endswith('g'):
                total_memory_gb = int(memory_str[:-1])
            elif memory_str.endswith('m'):
                total_memory_gb = int(memory_str[:-1]) / 1024
            else:
                total_memory_gb = 64  # Default fallback
        except (ValueError, AttributeError):
            total_memory_gb = 64  # Default fallback
        
        memory_based_threads = max(1, math.ceil(total_memory_gb / 32))
        # Cap by CPU count
        thread_count = min(CPU_COUNT, memory_based_threads, MAX_WORKERS)
    else:
        # For CPU-bound tasks, use more threads
        thread_count = min(MAX_WORKERS, CPU_COUNT * 2)
    
    logger.info(f"Using {thread_count} threads for {'memory-intensive' if memory_intensive else 'CPU-bound'} task")
    return thread_count

def parallel_map(func: Callable, items: Iterable, 
                 max_workers: Optional[int] = None, 
                 memory_intensive: bool = False,
                 show_progress: bool = True,
                 chunk_size: Optional[int] = None,
                 use_processes: bool = False,
                 **kwargs) -> List:
    """
    Apply a function to items in parallel using a thread or process pool.
    
    Args:
        func (Callable): Function to apply to each item
        items (Iterable): Items to process
        max_workers (int, optional): Maximum number of workers
        memory_intensive (bool): Whether the task is memory-intensive
        show_progress (bool): Whether to show progress
        chunk_size (int, optional): Size of chunks for processing
        use_processes (bool): Whether to use processes instead of threads
        **kwargs: Additional arguments to pass to the function
        
    Returns:
        List: Results for each item
    """
    # Convert items to list if it's not already
    items_list = list(items)
    n_items = len(items_list)
    
    if n_items == 0:
        return []
    
    # Determine number of workers
    if max_workers is None:
        max_workers = get_optimal_thread_count(memory_intensive)
    
    # For very small tasks, just use sequential processing
    if n_items <= 2:
        logger.info(f"Processing {n_items} items sequentially")
        if kwargs:
            partial_func = partial(func, **kwargs)
            return [partial_func(item) for item in items_list]
        else:
            return [func(item) for item in items_list]
    
    # Determine chunk size
    if chunk_size is None:
        chunk_size = max(1, min(100, n_items // (max_workers * 2)))
    
    # Create a partial function if kwargs are provided
    if kwargs:
        partial_func = partial(func, **kwargs)
    else:
        partial_func = func
    
    # Choose executor class based on whether to use processes or threads
    executor_class = concurrent.futures.ProcessPoolExecutor if use_processes else concurrent.futures.ThreadPoolExecutor
    
    logger.info(f"Processing {n_items} items with {max_workers} {'processes' if use_processes else 'threads'} "
                f"(chunk_size={chunk_size})")
    
    # Process items in parallel
    results = []
    start_time = time.time()
    last_log_time = start_time
    items_processed = 0
    
    with executor_class(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {executor.submit(partial_func, item): i for i, item in enumerate(items_list)}
        
        # Wait for each task to complete and get results
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results.append((idx, result))
                
                # Update progress
                items_processed += 1
                if show_progress and (time.time() - last_log_time > 5 or items_processed == n_items):
                    progress = items_processed / n_items * 100
                    elapsed = time.time() - start_time
                    rate = items_processed / elapsed if elapsed > 0 else 0
                    estimated_total = n_items / rate if rate > 0 else 0
                    remaining = estimated_total - elapsed if rate > 0 else 0
                    
                    logger.info(f"Progress: {items_processed}/{n_items} ({progress:.1f}%) - "
                                f"Rate: {rate:.1f} items/sec - "
                                f"Elapsed: {elapsed:.1f}s - "
                                f"Remaining: {remaining:.1f}s")
                    last_log_time = time.time()
                
            except Exception as e:
                logger.error(f"Error processing item {idx}: {e}")
                results.append((idx, None))
    
    # Sort results by index
    sorted_results = [result for _, result in sorted(results, key=lambda x: x[0])]
    
    total_time = time.time() - start_time
    logger.info(f"Completed {n_items} items in {total_time:.1f}s (avg: {n_items/total_time:.1f} items/sec)")
    
    return sorted_results

def parallel_chunked_task(func: Callable, chunks: List, 
                         max_workers: Optional[int] = None,
                         use_processes: bool = True,
                         **kwargs) -> List:
    """
    Process chunks of data in parallel using multiple processes.
    
    Args:
        func (Callable): Function to apply to each chunk
        chunks (List): Chunks of data to process
        max_workers (int, optional): Maximum number of workers
        use_processes (bool): Whether to use processes instead of threads
        **kwargs: Additional arguments to pass to the function
        
    Returns:
        List: Results for each chunk
    """
    return parallel_map(func, chunks, max_workers=max_workers, 
                        memory_intensive=True, use_processes=use_processes, 
                        **kwargs)

def split_into_chunks(items: List, chunk_size: Optional[int] = None, 
                     num_chunks: Optional[int] = None) -> List[List]:
    """
    Split a list into chunks.
    
    Args:
        items (List): Items to split
        chunk_size (int, optional): Size of each chunk
        num_chunks (int, optional): Number of chunks
        
    Returns:
        List[List]: List of chunks
    """
    n_items = len(items)
    
    if n_items == 0:
        return []
    
    # If neither chunk_size nor num_chunks is specified, determine based on CPU count
    if chunk_size is None and num_chunks is None:
        num_chunks = min(n_items, get_optimal_thread_count())
    
    # If chunk_size is specified, calculate number of chunks
    if chunk_size is not None:
        num_chunks = math.ceil(n_items / chunk_size)
    # If num_chunks is specified, calculate chunk size
    elif num_chunks is not None:
        chunk_size = math.ceil(n_items / num_chunks)
    
    # Create chunks
    chunks = []
    for i in range(0, n_items, chunk_size):
        chunks.append(items[i:i + chunk_size])
    
    logger.info(f"Split {n_items} items into {len(chunks)} chunks (avg size: {n_items/len(chunks):.1f})")
    
    return chunks

def run_with_thread_limit(func: Callable, limit: int = None, **kwargs) -> Any:
    """
    Run a function with a limited number of threads.
    
    Args:
        func (Callable): Function to run
        limit (int): Maximum number of threads (default: based on hardware)
        **kwargs: Arguments to pass to the function
        
    Returns:
        Any: Result of the function
    """
    import threadpoolctl
    
    # Determine thread limit if not specified
    if limit is None:
        limit = get_optimal_thread_count()
    
    logger.info(f"Running with thread limit: {limit}")
    
    # Run the function with thread limit
    with threadpoolctl.threadpool_limits(limits=limit, user_api='blas'):
        result = func(**kwargs)
    
    return result

def optimize_threadpool_settings(jvm_task: bool = False) -> None:
    """
    Optimize thread pool settings for various libraries.
    
    Args:
        jvm_task (bool): Whether the task involves JVM (e.g., H2O, Spark)
    """
    # Get optimal thread count
    thread_count = get_optimal_thread_count(memory_intensive=jvm_task)
    
    # Set OpenMP thread limit
    os.environ["OMP_NUM_THREADS"] = str(thread_count)
    logger.info(f"Set OMP_NUM_THREADS={thread_count}")
    
    # Set MKL thread limit
    os.environ["MKL_NUM_THREADS"] = str(thread_count)
    logger.info(f"Set MKL_NUM_THREADS={thread_count}")
    
    # Set OPENBLAS thread limit
    os.environ["OPENBLAS_NUM_THREADS"] = str(thread_count)
    logger.info(f"Set OPENBLAS_NUM_THREADS={thread_count}")
    
    # Set numpy thread limit
    try:
        import numpy as np
        np.set_num_threads(thread_count)
        logger.info(f"Set numpy.set_num_threads({thread_count})")
    except (ImportError, AttributeError):
        pass
    
    # Set scikit-learn thread limit
    try:
        from sklearn.utils import parallel_backend
        logger.info(f"Configure sklearn to use {thread_count} threads when possible")
    except ImportError:
        pass
    
    # Set pandas thread limit
    try:
        import pandas as pd
        pd.options.compute.use_bottleneck = True
        pd.options.compute.use_numexpr = True
        logger.info(f"Configured pandas to use bottleneck and numexpr")
    except (ImportError, AttributeError):
        pass
    
    # For JVM tasks, limit JVM threads
    if jvm_task:
        # Spark
        os.environ["SPARK_WORKER_CORES"] = str(thread_count)
        logger.info(f"Set SPARK_WORKER_CORES={thread_count}")
        
        # H2O
        os.environ["H2O_NTHREADS"] = str(thread_count)
        logger.info(f"Set H2O_NTHREADS={thread_count}")
        
        # General JVM
        os.environ["_JAVA_OPTIONS"] = f"-XX:ParallelGCThreads={thread_count // 2} -XX:ConcGCThreads={thread_count // 4}"
        logger.info(f"Set JVM garbage collection threads: ParallelGCThreads={thread_count // 2}, ConcGCThreads={thread_count // 4}")

if __name__ == "__main__":
    # Test threading utilities
    logger.info(f"CPU count: {CPU_COUNT}")
    logger.info(f"Max workers: {MAX_WORKERS}")
    
    # Test optimal thread count
    thread_count = get_optimal_thread_count()
    logger.info(f"Optimal thread count: {thread_count}")
    
    memory_thread_count = get_optimal_thread_count(memory_intensive=True)
    logger.info(f"Memory-intensive thread count: {memory_thread_count}")
    
    # Test optimize_threadpool_settings
    optimize_threadpool_settings()
    
    # Test parallel_map
    def test_func(x, multiplier=1):
        time.sleep(0.01)  # Simulate work
        return x * multiplier
    
    test_items = list(range(100))
    
    # Test with threads
    thread_results = parallel_map(test_func, test_items, multiplier=2, show_progress=True)
    logger.info(f"Thread results (first 5): {thread_results[:5]}")
    
    # Test with processes
    process_results = parallel_map(test_func, test_items, multiplier=3, use_processes=True, show_progress=True)
    logger.info(f"Process results (first 5): {process_results[:5]}")
    
    # Test split_into_chunks
    chunks = split_into_chunks(test_items, chunk_size=10)
    logger.info(f"Chunks (first 2): {chunks[:2]}")
    
    # Test run_with_thread_limit
    try:
        import numpy as np
        result = run_with_thread_limit(lambda: np.dot(np.random.random((1000, 1000)), np.random.random((1000, 1000))))
        logger.info(f"Matrix multiplication completed with shape {result.shape}")
    except ImportError:
        logger.warning("NumPy not available, skipping matrix multiplication test")