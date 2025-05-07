# H2O Sparkling Water Testing

This directory contains scripts for testing H2O Sparkling Water with Java 11, which is required for compatibility.

## Java Version Compatibility

H2O Sparkling Water has specific Java version requirements:
- Only Java versions 8-17 are supported
- Java 21 will not work with H2O Sparkling Water
- We have confirmed that Java 11 works correctly

## Test Scripts

1. **test_h2o_sparkling_minimal.py**
   - Minimal test that verifies basic H2O Sparkling Water functionality
   - Creates a simple Spark session and H2O Context
   - Minimal data processing - focuses on verifying setup works

2. **test_h2o_sparkling_java11.py**
   - More comprehensive test with synthetic dataset using Java 11
   - Tests data conversion between Spark and H2O
   - Better for functional testing after environment is confirmed working

3. **test_h2o_sparkling_java17.py**
   - Test for H2O Sparkling Water with Java 17
   - Verifies compatibility with the latest supported Java version
   - Requires Java 17 to be installed with module options

4. **test_h2o_java17_simple.py**
   - Ultra-minimal test for H2O Sparkling Water with Java 17
   - Only checks version and package compatibility
   - Fast verification without data processing

## Environment Setup

1. **setup_java11_env.sh**
   - Sets up Java 11 in the current shell
   - Must be sourced: `source setup_java11_env.sh`
   - Sets JAVA_HOME and PATH to use Java 11

2. **setup_java17_env.sh**
   - Sets up Java 17 in the current shell
   - Must be sourced: `source setup_java17_env.sh`
   - Sets JAVA_HOME and PATH to use Java 17

## Virtual Environment

For a complete test environment, you can create a dedicated venv:

```bash
# Create and activate a virtual environment
python -m venv test_env_h2o
source test_env_h2o/bin/activate

# Set up Java 11
source setup_java11_env.sh

# Install required packages
pip install setuptools numpy pandas scikit-learn
pip install h2o==3.46.0.7 pyspark==3.5.0
pip install h2o-pysparkling-3.5==3.46.0.6.post1

# Run tests with either Java 11 or Java 17
source setup_java11_env.sh
python test_h2o_sparkling_java11.py

# OR

source setup_java17_env.sh
python test_h2o_sparkling_java17.py
```

## Troubleshooting

1. **Java Version Issues**
   - Error: "Unsupported Java version"
   - Solution: Make sure you're using Java 11 by running `source setup_java11_env.sh`

2. **Memory Issues**
   - If the tests timeout or fail due to OOM (Out Of Memory) errors
   - Solution: Reduce the dataset size or increase Spark memory settings

3. **Package Import Issues**
   - Error: "No module named 'pkg_resources'"
   - Solution: `pip install setuptools`

## Notes

- H2O Sparkling Water tests can be slow due to initialization overhead
- The tests may time out in limited environments but still work
- For production systems, ensure adequate memory is allocated
- For Numerai use cases, configure with more memory in the settings.py