# Deployment Guide for Clinical CDSS Readmission Prediction System

This guide provides instructions for deploying the Clinical CDSS Readmission Prediction System using Docker.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (version 20.10.0 or higher)
- [Docker Compose](https://docs.docker.com/compose/install/) (version 2.0.0 or higher)
- At least 4GB of RAM and 2 CPU cores
- 10GB of free disk space

## Local Deployment

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/clinical-cdss-readmission.git
cd clinical-cdss-readmission
```

### 2. Build and Start the Containers

```bash
docker-compose up -d
```

This command will:
- Build the Docker images for the API and web services
- Start the containers in detached mode
- Map port 8000 for the API and port 80 for the web interface

### 3. Verify Deployment

Check if the containers are running:

```bash
docker-compose ps
```

Access the web interface:
- Open your browser and navigate to `http://localhost`

Access the API documentation:
- Open your browser and navigate to `http://localhost/api/docs`

### 4. Stopping the Application

```bash
docker-compose down
```

## Production Deployment

For production deployment, additional steps are recommended:

### 1. Configure Environment Variables

Create a `.env` file in the project root:

```
ENVIRONMENT=production
LOG_LEVEL=WARNING
MAX_WORKERS=8
```

### 2. Use HTTPS

For production, you should use HTTPS. Update the `nginx.conf` file to include SSL configuration and obtain SSL certificates using Let's Encrypt.

### 3. Set Up Monitoring

Configure Prometheus and Grafana for monitoring:

```bash
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
```

## Cloud Deployment Options

### AWS Elastic Container Service (ECS)

1. Create an ECR repository for your images
2. Push your Docker images to ECR
3. Create an ECS cluster
4. Define task definitions for your services
5. Create ECS services to run your tasks
6. Set up an Application Load Balancer

### Google Cloud Run

1. Push your Docker images to Google Container Registry
2. Deploy the API service to Cloud Run
3. Deploy the web service to Cloud Run
4. Configure Cloud Run services to communicate with each other

### Azure Container Instances

1. Push your Docker images to Azure Container Registry
2. Deploy the containers to Azure Container Instances
3. Set up Azure Application Gateway for routing

## Scaling Considerations

- The API service can be scaled horizontally by increasing the number of replicas
- Consider using a managed database service for storing prediction logs
- Use a CDN for serving static assets in the web service
- Implement rate limiting for the API to prevent abuse

## Troubleshooting

### Container Fails to Start

Check the logs:

```bash
docker-compose logs api
docker-compose logs web
```

### API Returns 500 Errors

Ensure the model files are correctly mounted:

```bash
docker-compose exec api ls -la /app/models
```

### Web Interface Not Loading

Check if the API is accessible from the web container:

```bash
docker-compose exec web curl api:8000/health
```

## Backup and Restore

### Backing Up Model Files

```bash
docker-compose exec api tar -czvf /tmp/models_backup.tar.gz /app/models
docker cp $(docker-compose ps -q api):/tmp/models_backup.tar.gz ./models_backup.tar.gz
```

### Restoring Model Files

```bash
docker cp ./models_backup.tar.gz $(docker-compose ps -q api):/tmp/
docker-compose exec api tar -xzvf /tmp/models_backup.tar.gz -C /
```
