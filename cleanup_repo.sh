#!/bin/bash
# Script to clean up the repository by removing temporary files and organizing the structure

echo "Starting repository cleanup..."

# Remove temporary debugging scripts
rm -f check_numerapi.py
rm -f check_symbols.py
rm -f examine_targets.py
rm -f examine_yiedl.py
rm -f implementation_plan.md

# Remove temporary log files
rm -f *.log
rm -f scripts/*.log
rm -f scripts/h2ologs/*.log

# Clean up .ipynb_checkpoints
find . -name ".ipynb_checkpoints" -type d -exec rm -rf {} +

# Clean up __pycache__ directories
find . -name "__pycache__" -type d -exec rm -rf {} +

# Clean up temporary notebook outputs
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebooks/*.ipynb

# Ensure directory structure is consistent
mkdir -p data/processed
mkdir -p data/submissions
mkdir -p models/comparison
mkdir -p reports/figures

# Make sure all Python files have proper permissions
find . -name "*.py" -exec chmod +x {} \;

# Make sure all shell scripts have proper permissions
find . -name "*.sh" -exec chmod +x {} \;

echo "Repository cleanup complete!"