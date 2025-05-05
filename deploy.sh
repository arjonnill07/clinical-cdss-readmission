#!/bin/bash

# Bash script for deploying the Clinical CDSS Readmission Prediction System

# Display banner
echo "=================================================="
echo "Clinical CDSS Readmission Prediction System Deploy"
echo "=================================================="
echo ""

# Check if Docker is installed
if command -v docker &> /dev/null; then
    docker_version=$(docker --version)
    echo "✓ Docker is installed: $docker_version"
else
    echo "✗ Docker is not installed or not in PATH. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if command -v docker-compose &> /dev/null; then
    compose_version=$(docker-compose --version)
    echo "✓ Docker Compose is installed: $compose_version"
else
    echo "✗ Docker Compose is not installed or not in PATH. Please install Docker Compose first."
    exit 1
fi

# Check if models directory exists and has files
if [ -f "models/model.pkl" ]; then
    echo "✓ Model files found"
else
    echo "✗ Model files not found. Please run the training pipeline first."
    read -p "Do you want to run the model training pipeline now? (y/n) " train_model
    if [ "$train_model" = "y" ]; then
        echo "Running model training pipeline..."
        python src/data/prepare.py
        python src/features/build_features.py
        python src/models/train_model.py
        python src/models/evaluate_model.py
    else
        echo "Deployment aborted. Please run the training pipeline and try again."
        exit 1
    fi
fi

# Build and start the containers
echo ""
echo "Building and starting containers..."
docker-compose up --build -d

# Check if containers are running
sleep 5
containers=$(docker-compose ps)
if echo "$containers" | grep -q "Up"; then
    echo "✓ Containers are running successfully"
    
    # Get container IPs
    api_url="http://localhost:8000"
    web_url="http://localhost"
    
    echo ""
    echo "Application deployed successfully!"
    echo "API URL: $api_url"
    echo "Web URL: $web_url"
    echo ""
    echo "API Documentation: $api_url/docs"
    echo "Health Check: $api_url/health"
    
    # Open web interface in browser
    read -p "Do you want to open the web interface in your browser? (y/n) " open_browser
    if [ "$open_browser" = "y" ]; then
        if command -v xdg-open &> /dev/null; then
            xdg-open $web_url
        elif command -v open &> /dev/null; then
            open $web_url
        else
            echo "Could not open browser automatically. Please open $web_url manually."
        fi
    fi
else
    echo "✗ There was an issue starting the containers. Please check the logs:"
    docker-compose logs
fi

echo ""
echo "To stop the application, run: docker-compose down"
echo "To view logs, run: docker-compose logs"
echo "For more information, see the deployment guide in docs/deployment.md"
