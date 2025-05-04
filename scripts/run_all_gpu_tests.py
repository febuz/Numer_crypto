#!/usr/bin/env python3
"""
Main script to run all GPU tests.
This script provides a command-line interface to run the GPU tests.
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

def run_test(test_path, args=None):
    """Run a specific test with optional arguments"""
    cmd = [sys.executable, str(test_path)]
    if args:
        cmd.extend(args)
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"Error running test: {test_path}")
        return False

def setup_java(version=11):
    """Set up Java environment based on version"""
    if version == 11:
        script_path = project_root / "scripts" / "setup_java11_env.sh"
    elif version == 17:
        script_path = project_root / "scripts" / "setup_java17_env.sh"
    else:
        print(f"Unsupported Java version: {version}")
        return False
    
    if not script_path.exists():
        print(f"Java setup script not found: {script_path}")
        return False
    
    try:
        # Source the setup script (this requires a shell)
        subprocess.run(f"source {script_path}", shell=True, executable="/bin/bash")
        return True
    except subprocess.SubprocessError as e:
        print(f"Error setting up Java {version}: {e}")
        return False

def main():
    """Main function to run the GPU tests"""
    parser = argparse.ArgumentParser(description="Run GPU performance tests")
    parser.add_argument("--test", choices=["peak", "multi-gpu", "java-comparison", "all"], 
                      default="all", help="Specify which test to run")
    parser.add_argument("--java-version", type=int, choices=[11, 17], default=11,
                      help="Java version to use")
    parser.add_argument("--rows", type=int, default=100000,
                      help="Number of rows in the dataset")
    parser.add_argument("--cols", type=int, default=20,
                      help="Number of features in the dataset")
    parser.add_argument("--gpu-id", type=int, default=0,
                      help="GPU ID to use (for single-GPU tests)")
    
    args = parser.parse_args()
    
    # Set up Java environment
    if not setup_java(args.java_version):
        print("Failed to set up Java environment. Exiting.")
        return 1
    
    # Ensure reports directory exists
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    tests_dir = project_root / "tests" / "performance"
    success = True
    
    # Run requested tests
    if args.test in ["peak", "all"]:
        print("\n" + "="*80)
        print("RUNNING PEAK GPU UTILIZATION TEST")
        print("="*80)
        test_path = tests_dir / "test_peak_gpu.py"
        success = success and run_test(test_path)
    
    if args.test in ["multi-gpu", "all"]:
        print("\n" + "="*80)
        print("RUNNING MULTI-GPU H2O TEST")
        print("="*80)
        test_path = tests_dir / "test_multi_gpu_h2o.py"
        test_args = [
            "--rows", str(args.rows),
            "--cols", str(args.cols),
            "--java-version", str(args.java_version),
            "--output-dir", str(reports_dir)
        ]
        success = success and run_test(test_path, test_args)
    
    if args.test in ["java-comparison", "all"]:
        print("\n" + "="*80)
        print("RUNNING JAVA COMPARISON TEST")
        print("="*80)
        test_path = tests_dir / "test_java_gpu_comparison.py"
        test_args = [
            "--rows", str(args.rows),
            "--cols", str(args.cols),
            "--gpu-id", str(args.gpu_id),
            "--output-dir", str(reports_dir)
        ]
        success = success and run_test(test_path, test_args)
    
    # Return summary
    if success:
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Results saved to: {reports_dir}")
        return 0
    else:
        print("\n" + "="*80)
        print("SOME TESTS FAILED")
        print("="*80)
        return 1

if __name__ == "__main__":
    sys.exit(main())