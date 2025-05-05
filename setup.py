from setuptools import find_packages, setup

setup(
    name="clinical-cdss-readmission",
    version="0.1.0",
    description="AI-Powered Clinical Decision Support System for Predicting Patient Readmissions",
    author="Arjon Golder",
    author_email="arjon16@cse.pstu.ac.bd",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "dvc>=2.0.0",
        "apache-airflow>=2.2.0",
        "transformers>=4.5.0",
        "torch>=1.9.0",
        "xgboost>=1.4.0",
        "shap>=0.39.0",
        "mlflow>=1.20.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
    ],
    python_requires=">=3.8",
)
