#!/usr/bin/env python3
"""
Validate Submissions for Numerai Crypto

This script analyzes all submission files to determine the best one for Numerai Crypto competition.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
submissions_dir = project_root / "data" / "submissions"

print(f"Analyzing submission files in {submissions_dir}")

# Find all submission CSV files
csv_files = list(submissions_dir.glob("*.csv"))
print(f"Found {len(csv_files)} submission files")

# Analyze each file
results = []

for file_path in csv_files:
    try:
        print(f"Analyzing {file_path.name}...")
        df = pd.read_csv(file_path)
        
        if 'id' not in df.columns or 'prediction' not in df.columns:
            print(f"Warning: File {file_path.name} doesn't have required columns")
            continue
        
        # Calculate statistics
        stats = {
            'file': str(file_path),
            'name': file_path.name,
            'size': os.path.getsize(file_path),
            'rows': len(df),
            'mean': float(df['prediction'].mean()),
            'std': float(df['prediction'].std()),
            'min': float(df['prediction'].min()),
            'max': float(df['prediction'].max()),
            'median': float(df['prediction'].median()),
            'null_count': int(df['prediction'].isnull().sum()),
            # Calculate frequency distribution (10 buckets)
            'distribution': np.histogram(df['prediction'], bins=10, range=(0, 1))[0].tolist()
        }
        
        results.append(stats)
        print(f"  Rows: {stats['rows']}, Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
        
    except Exception as e:
        print(f"Error analyzing {file_path.name}: {e}")

# Sort results by quality metrics
# For Numerai, a good submission typically has:
# - Reasonable standard deviation (not too high or too low)
# - Mean around 0.5
# - Good distribution spread
if results:
    print("\nRanking submission files...")
    
    for result in results:
        # Calculate quality score
        # 1. Closeness of mean to 0.5 (ideal is 0.5)
        mean_score = 1.0 - abs(result['mean'] - 0.5) * 2
        
        # 2. Standard deviation in a good range (ideal around 0.15-0.25)
        std_dev = result['std']
        if std_dev < 0.05:
            std_score = std_dev / 0.15  # Too low
        elif std_dev > 0.4:
            std_score = 2 - std_dev / 0.4  # Too high
        else:
            # Optimal at 0.2
            std_score = 1.0 - abs(std_dev - 0.2) * 2
        
        # 3. Distribution shape (even distribution is better)
        dist = np.array(result['distribution'])
        dist_norm = dist / dist.sum()
        # Calculate entropy (higher is better)
        entropy = -np.sum(dist_norm * np.log2(dist_norm + 1e-10))
        # Max entropy for 10 bins is log2(10) ≈ 3.32
        dist_score = entropy / 3.32
        
        # Combine scores with weights
        quality_score = (mean_score * 0.3) + (std_score * 0.5) + (dist_score * 0.2)
        result['quality_score'] = quality_score
    
    # Sort by quality score
    results.sort(key=lambda x: x['quality_score'], reverse=True)
    
    # Print top results
    print("\nBest submission files:")
    for i, result in enumerate(results[:3]):
        print(f"{i+1}. {result['name']}")
        print(f"   Quality Score: {result['quality_score']:.4f}")
        print(f"   Mean: {result['mean']:.4f}, Std: {result['std']:.4f}")
        print(f"   Rows: {result['rows']}, Size: {result['size'] / 1024:.1f} KB")
        print()
    
    # Save analysis results
    output_file = submissions_dir / f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Analysis results saved to {output_file}")
    
    # Print submission command for the best file
    best_file = results[0]['file']
    print("\nTo submit the best file to Numerai, run:")
    print(f"python -c \"from numer_crypto.data.retrieval import NumeraiDataRetriever; NumeraiDataRetriever().submit_predictions('{best_file}', 'crypto')\"")

else:
    print("No valid submission files found")