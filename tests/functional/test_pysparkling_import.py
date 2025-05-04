#!/usr/bin/env python3
"""
Test importing PySparkling modules
This script attempts to identify the correct import paths for PySparkling
"""
import sys
import os
import importlib
import pkgutil

print("Python version:", sys.version)
print("Python executable:", sys.executable)

def check_module_exists(module_name):
    """Check if a module exists without importing it"""
    try:
        importlib.util.find_spec(module_name)
        return True
    except (ImportError, ModuleNotFoundError):
        return False

# List all installed packages for h2o
print("\nInstalled packages:")
for package in sorted([p.name for p in pkgutil.iter_modules()]):
    if "h2o" in package or "spark" in package or "h2o_" in package:
        print(f"- {package}")

# Try importing basic modules
print("\nTrying to import basic modules:")
try:
    import h2o
    print("import h2o: SUCCESS")
    print(f"H2O version: {h2o.__version__}")
except ImportError as e:
    print(f"import h2o: FAILED - {e}")

try:
    import pyspark
    print("import pyspark: SUCCESS")
    print(f"PySpark version: {pyspark.__version__}")
except ImportError as e:
    print(f"import pyspark: FAILED - {e}")

# Try various pysparkling import paths
print("\nTrying different pysparkling import paths:")

try:
    import pysparkling
    print("import pysparkling: SUCCESS")
    print(f"PySparkling path: {pysparkling.__file__}")
except ImportError as e:
    print(f"import pysparkling: FAILED - {e}")

try:
    import h2o_pysparkling_3_5
    print("import h2o_pysparkling_3_5: SUCCESS")
    print(f"h2o_pysparkling_3_5 path: {h2o_pysparkling_3_5.__file__}")
except ImportError as e:
    print(f"import h2o_pysparkling_3_5: FAILED - {e}")

try:
    from pysparkling import H2OContext
    print("from pysparkling import H2OContext: SUCCESS")
except ImportError as e:
    print(f"from pysparkling import H2OContext: FAILED - {e}")

# List the site-packages directory
site_packages = [p for p in sys.path if 'site-packages' in p][0]
print(f"\nContents of site-packages directory ({site_packages}):")
h2o_packages = [f for f in os.listdir(site_packages) if 'h2o' in f.lower() or 'spark' in f.lower()]
for pkg in sorted(h2o_packages):
    print(f"- {pkg}")

# Look for specific jar files
print("\nLooking for H2O jar files:")
for root, _, files in os.walk(site_packages):
    for file in files:
        if file.endswith('.jar') and ('h2o' in file.lower() or 'spark' in file.lower()):
            print(f"- {os.path.join(root, file)}")