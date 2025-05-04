#!/usr/bin/env python
"""
Script to check Polars installation and performance.
"""
import sys
import time
import platform

print(f"Python version: {platform.python_version()}")

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
    
    # Try to import Polars
    import polars as pl
    print(f"Polars version: {pl.__version__}")
    
    # Generate some test data
    print("\nGenerating test data...")
    n_rows = 100000
    data = {
        'A': np.random.rand(n_rows),
        'B': np.random.rand(n_rows),
        'C': np.random.randint(0, 100, size=n_rows)
    }
    
    # Test Polars DataFrame
    print("\nTesting Polars operations:")
    start_time = time.time()
    df = pl.DataFrame(data)
    print(f"  DataFrame shape: {df.shape}")
    
    # Basic operations
    print("  Performing operations:")
    result = df.group_by('C').agg([
        pl.col('A').mean().alias('A_mean'),
        pl.col('B').sum().alias('B_sum')
    ])
    
    elapsed = time.time() - start_time
    print(f"  Polars operations completed in {elapsed:.4f} seconds")
    print(f"  Result shape: {result.shape}")
    print(f"  First few rows of result:\n{result.head()}")
    
    # Try to import pandas for comparison
    try:
        import pandas as pd
        print(f"\nPandas version: {pd.__version__}")
        
        # Test pandas with the same data
        print("Testing pandas operations:")
        start_time = time.time()
        pdf = pd.DataFrame(data)
        
        # Same operations in pandas
        presult = pdf.groupby('C').agg({'A': 'mean', 'B': 'sum'})
        
        pd_elapsed = time.time() - start_time
        print(f"  Pandas operations completed in {pd_elapsed:.4f} seconds")
        
        if elapsed < pd_elapsed:
            speedup = pd_elapsed / elapsed
            print(f"  Polars is {speedup:.2f}x faster than pandas")
        else:
            slowdown = elapsed / pd_elapsed
            print(f"  Polars is {slowdown:.2f}x slower than pandas")
            
    except ImportError:
        print("Pandas not installed, skipping comparison")
        
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Error during test: {e}")