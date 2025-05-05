# AI-Powered Clinical Decision Support System for Predicting Patient Readmissions

## Project Overview
This project implements an end-to-end AI-powered clinical decision support system for predicting 30-day readmissions for diabetic patients. The system uses both structured EHR data and unstructured clinical notes while ensuring explainability and compliance with regulations like GDPR.

## Tech Stack
- **Data Versioning**: DVC + AWS S3
- **Orchestration**: Apache Airflow
- **Model Training**: Scikit-learn, XGBoost, HuggingFace Transformers
- **Experiment Tracking**: MLflow
- **Deployment**: FastAPI + Docker + Kubernetes (EKS)
- **Monitoring**: Prometheus + Grafana + Evidently AI
- **CI/CD**: GitHub Actions

## Project Structure
```
clinical-cdss-readmission/
├── .github/                    # GitHub Actions workflows
├── airflow/                    # Airflow DAGs and plugins
├── data/                       # Data directory (managed by DVC)
│   ├── raw/                    # Raw data
│   ├── processed/              # Processed data
│   └── features/               # Feature engineering outputs
├── docs/                       # Documentation
├── kubernetes/                 # Kubernetes manifests
├── models/                     # Model artifacts
├── notebooks/                  # Jupyter notebooks for exploration
├── src/                        # Source code
│   ├── data/                   # Data processing scripts
│   ├── features/               # Feature engineering
│   ├── models/                 # Model training and evaluation
│   ├── monitoring/             # Monitoring and alerting
│   └── api/                    # FastAPI application
├── tests/                      # Unit and integration tests
├── .dvcignore                  # DVC ignore file
├── .gitignore                  # Git ignore file
├── dvc.yaml                    # DVC pipeline definition
├── Dockerfile                  # Docker image definition
├── requirements.txt            # Python dependencies
└── setup.py                    # Package setup
```

## Getting Started
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Initialize DVC: `dvc init`
4. Configure AWS S3 remote: `dvc remote add -d s3remote s3://your-bucket/path`

## Data Pipeline
The data pipeline is orchestrated using Apache Airflow and versioned with DVC.

## Model Training
Models are trained using Scikit-learn, XGBoost, and HuggingFace Transformers. Experiments are tracked with MLflow.

## Deployment
The model is deployed as a FastAPI application in Docker containers orchestrated by Kubernetes.

## Monitoring
Model performance and data drift are monitored using Prometheus, Grafana, and Evidently AI.

## License
[Your License]
