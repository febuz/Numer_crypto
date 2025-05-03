from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="numer_crypto",
    version="0.1.0",
    author="Numerai Crypto Team",
    author_email="your.email@example.com",
    description="Numerai Crypto prediction models using LGBM, H2O XGBoost, Sparkling Water, and PySpark",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/numer_crypto",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "numer-crypto=numer_crypto.scripts.run:main",
        ],
    },
)