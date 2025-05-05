"""
Airflow DAG for orchestrating the Clinical CDSS Readmission prediction pipeline.

This DAG:
1. Downloads and prepares the data
2. Performs feature engineering
3. Trains the model
4. Evaluates the model
5. Deploys the model to the API service

Theory:
- Workflow orchestration ensures reproducibility and reliability
- DAGs (Directed Acyclic Graphs) represent task dependencies
- Airflow provides scheduling, monitoring, and error handling
- Parameterization allows for flexible pipeline execution
- Sensors can monitor for new data or trigger events
"""

from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.models import Variable
from airflow.utils.dates import days_ago

# Define default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['your.email@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'readmission_prediction_pipeline',
    default_args=default_args,
    description='End-to-end pipeline for readmission prediction',
    schedule_interval=timedelta(days=1),  # Daily run
    start_date=days_ago(1),
    tags=['healthcare', 'ml', 'readmission'],
    catchup=False,
)

# Define the project directory
# In a production environment, this would be set as an Airflow Variable
project_dir = "{{ var.value.project_dir }}"
if not project_dir:
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

# Task 1: Data Preparation
data_preparation = BashOperator(
    task_id='data_preparation',
    bash_command=f'cd {project_dir} && python src/data/prepare.py',
    dag=dag,
)

# Task 2: Check if processed data exists
check_processed_data = FileSensor(
    task_id='check_processed_data',
    filepath=f'{project_dir}/data/processed/train.csv',
    poke_interval=60,  # Check every minute
    timeout=60 * 10,  # Timeout after 10 minutes
    mode='poke',
    dag=dag,
)

# Task 3: Feature Engineering
feature_engineering = BashOperator(
    task_id='feature_engineering',
    bash_command=f'cd {project_dir} && python src/features/build_features.py',
    dag=dag,
)

# Task 4: Check if feature matrices exist
check_features = FileSensor(
    task_id='check_features',
    filepath=f'{project_dir}/data/features/X_train.csv',
    poke_interval=60,
    timeout=60 * 10,
    mode='poke',
    dag=dag,
)

# Task 5: Model Training
model_training = BashOperator(
    task_id='model_training',
    bash_command=f'cd {project_dir} && python src/models/train_model.py',
    dag=dag,
)

# Task 6: Check if model exists
check_model = FileSensor(
    task_id='check_model',
    filepath=f'{project_dir}/models/model.pkl',
    poke_interval=60,
    timeout=60 * 10,
    mode='poke',
    dag=dag,
)

# Task 7: Model Evaluation
model_evaluation = BashOperator(
    task_id='model_evaluation',
    bash_command=f'cd {project_dir} && python src/models/evaluate_model.py',
    dag=dag,
)

# Task 8: Check if evaluation results exist
check_evaluation = FileSensor(
    task_id='check_evaluation',
    filepath=f'{project_dir}/models/evaluation.json',
    poke_interval=60,
    timeout=60 * 10,
    mode='poke',
    dag=dag,
)

# Task 9: Model Deployment
# In a real-world scenario, this would involve more complex steps
# such as building a Docker image, pushing it to a registry, and updating Kubernetes deployments
model_deployment = BashOperator(
    task_id='model_deployment',
    bash_command=f'echo "Deploying model to production" && '
                 f'cp {project_dir}/models/model.pkl {project_dir}/models/model_prod.pkl && '
                 f'echo "Model deployed successfully"',
    dag=dag,
)

# Define task dependencies
data_preparation >> check_processed_data >> feature_engineering >> check_features
check_features >> model_training >> check_model >> model_evaluation >> check_evaluation >> model_deployment
