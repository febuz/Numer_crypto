from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    # Filter out comments and empty lines
    requirements = [line.strip() for line in fh.read().splitlines() 
                   if line.strip() and not line.strip().startswith('#') 
                   and not line.strip().startswith('cudf') 
                   and not line.strip().startswith('cuml')
                   and not line.strip().startswith('cugraph')
                   and not line.strip().startswith('cuspatial')
                   and not line.strip().startswith('cupy')
                   and not line.strip().startswith('dask-cuda')
                   and not line.strip().startswith('rapids-4-spark')]

setup(
    name="numer_crypto",
    version="0.1.0",
    author="Numerai Crypto Team",
    author_email="e.h.hauwert@ehmac.nl",
    description="Numerai Crypto prediction models using LGBM, H2O XGBoost, Sparkling Water, and PySpark",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/febuz/Numer_crypto",
    packages=find_packages(include=['scripts', 'utils', 'models', 'config', 'data', 'feature']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "numer-crypto=scripts.run:main",
            "numer-gpu-test=scripts.run_all_gpu_tests:main",
            "numer-crypto-eda=feature.EDA.exploratory_analysis:main",
            "numer-crypto-features=feature.feature_engineering.polynomial_features:main",
        ],
    },
)