# AI-Powered Clinical Decision Support System for Predicting Patient Readmissions

## Project Overview
This project implements an end-to-end AI-powered clinical decision support system for predicting 30-day readmissions for diabetic patients. The system uses both structured EHR data and unstructured clinical notes while ensuring explainability and compliance with regulations like GDPR.

![System Architecture](docs/images/architecture.png)

## Key Features
- **Hybrid Data Processing**: Combines structured EHR data with unstructured clinical notes
- **Advanced ML Models**: Ensemble of traditional ML and deep learning models
- **Explainable AI**: SHAP-based explanations for model predictions
- **Regulatory Compliance**: Built with GDPR and HIPAA compliance in mind
- **Continuous Monitoring**: Detects data drift and model performance degradation
- **Scalable Architecture**: Kubernetes-based deployment for horizontal scaling
- **End-to-End MLOps**: From data ingestion to model deployment and monitoring

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

### Prerequisites
- Python 3.9+
- Docker and Docker Compose
- AWS CLI configured with appropriate permissions
- kubectl for Kubernetes deployment

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/clinical-cdss-readmission.git
   cd clinical-cdss-readmission
   ```

2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Initialize DVC
   ```bash
   dvc init
   ```

5. Configure AWS S3 remote for DVC
   ```bash
   dvc remote add -d s3remote s3://your-bucket/path
   ```

### Running the Pipeline

1. Run the data preparation step
   ```bash
   python src/data/prepare.py
   ```

2. Run the feature engineering step
   ```bash
   python src/features/build_features.py
   ```

3. Train the model
   ```bash
   python src/models/train_model.py
   ```

4. Evaluate the model
   ```bash
   python src/models/evaluate_model.py
   ```

5. Start the API locally
   ```bash
   uvicorn src.api.main:app --reload
   ```

### Running with Airflow

1. Set up Airflow
   ```bash
   export AIRFLOW_HOME=$(pwd)/airflow
   airflow db init
   airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com
   ```

2. Start Airflow
   ```bash
   airflow webserver -p 8080 &
   airflow scheduler &
   ```

3. Access Airflow UI at http://localhost:8080 and trigger the DAG

### Deployment

1. Build the Docker image
   ```bash
   docker build -t clinical-cdss-readmission:latest .
   ```

2. Push to your container registry
   ```bash
   docker tag clinical-cdss-readmission:latest your-registry/clinical-cdss-readmission:latest
   docker push your-registry/clinical-cdss-readmission:latest
   ```

3. Deploy to Kubernetes
   ```bash
   kubectl apply -f kubernetes/deployment.yaml
   ```

## Documentation

For more detailed documentation, please refer to:

- [Project Summary](docs/project_summary.md)
- [System Architecture](docs/architecture.md)
- [API Documentation](docs/api.md)
- [Model Documentation](docs/model.md)
- [Monitoring Guide](docs/monitoring.md)

## Testing

Run the tests with:

```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- UCI Machine Learning Repository for the Diabetes 130-US hospitals dataset
- The open-source community for the amazing tools and libraries used in this project
