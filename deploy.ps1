# PowerShell script for deploying the Clinical CDSS Readmission Prediction System

# Display banner
Write-Host "=================================================="
Write-Host "Clinical CDSS Readmission Prediction System Deploy"
Write-Host "=================================================="
Write-Host ""

# Check if Docker is installed
try {
    $dockerVersion = docker --version
    Write-Host "✓ Docker is installed: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Docker is not installed or not in PATH. Please install Docker first." -ForegroundColor Red
    exit 1
}

# Check if Docker Compose is installed
try {
    $composeVersion = docker-compose --version
    Write-Host "✓ Docker Compose is installed: $composeVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Docker Compose is not installed or not in PATH. Please install Docker Compose first." -ForegroundColor Red
    exit 1
}

# Check if models directory exists and has files
if (Test-Path -Path "models/model.pkl") {
    Write-Host "✓ Model files found" -ForegroundColor Green
} else {
    Write-Host "✗ Model files not found. Please run the training pipeline first." -ForegroundColor Red
    $trainModel = Read-Host "Do you want to run the model training pipeline now? (y/n)"
    if ($trainModel -eq "y") {
        Write-Host "Running model training pipeline..."
        python src/data/prepare.py
        python src/features/build_features.py
        python src/models/train_model.py
        python src/models/evaluate_model.py
    } else {
        Write-Host "Deployment aborted. Please run the training pipeline and try again."
        exit 1
    }
}

# Build and start the containers
Write-Host ""
Write-Host "Building and starting containers..."
docker-compose up --build -d

# Check if containers are running
Start-Sleep -Seconds 5
$containers = docker-compose ps
if ($containers -match "Up") {
    Write-Host "✓ Containers are running successfully" -ForegroundColor Green
    
    # Get container IPs
    $apiUrl = "http://localhost:8000"
    $webUrl = "http://localhost"
    
    Write-Host ""
    Write-Host "Application deployed successfully!"
    Write-Host "API URL: $apiUrl"
    Write-Host "Web URL: $webUrl"
    Write-Host ""
    Write-Host "API Documentation: $apiUrl/docs"
    Write-Host "Health Check: $apiUrl/health"
    
    # Open web interface in browser
    $openBrowser = Read-Host "Do you want to open the web interface in your browser? (y/n)"
    if ($openBrowser -eq "y") {
        Start-Process $webUrl
    }
} else {
    Write-Host "✗ There was an issue starting the containers. Please check the logs:" -ForegroundColor Red
    docker-compose logs
}

Write-Host ""
Write-Host "To stop the application, run: docker-compose down"
Write-Host "To view logs, run: docker-compose logs"
Write-Host "For more information, see the deployment guide in docs/deployment.md"
