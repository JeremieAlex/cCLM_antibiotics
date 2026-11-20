from setuptools import find_packages, setup

setup(
    name="cclm",
    version="1.0.0",
    description="Conditional chemical language model",
    packages=find_packages(include=["cclm", "cclm.*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "joblib>=1.3",
        "rdkit>=2023.9",
        "tqdm>=4.65",
        "tomli>=2.0; python_version < '3.11'",
    ],
)
