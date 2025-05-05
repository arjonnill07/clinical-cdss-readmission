# Deploying to Render

This guide provides step-by-step instructions for deploying the Clinical CDSS Readmission Prediction System to Render.

## Prerequisites

1. A [Render account](https://render.com/)
2. Git repository with your project code (GitHub, GitLab, or Bitbucket)

## Deployment Options

There are two ways to deploy to Render:

1. **Automatic deployment using Blueprint** (recommended)
2. **Manual deployment** of individual services

## Option 1: Automatic Deployment with Blueprint

### Step 1: Connect Your Repository

1. Log in to your Render account
2. Go to the Dashboard and click on "Blueprints" in the sidebar
3. Click "New Blueprint Instance"
4. Connect your Git repository (GitHub, GitLab, or Bitbucket)
5. Select the repository containing your project

### Step 2: Configure and Deploy

1. Render will automatically detect the `render.yaml` file in your repository
2. Review the services that will be created
3. Click "Apply" to start the deployment process
4. Render will create and deploy all services defined in the Blueprint

### Step 3: Access Your Application

Once deployment is complete:

1. Go to the Render Dashboard to see your services
2. Click on the `cdss-web` service to find the URL of your web interface
3. Your application is now live and accessible via the provided URL

## Option 2: Manual Deployment

If you prefer to deploy services individually:

### Step 1: Deploy the API Backend

1. Log in to your Render account
2. Click "New" and select "Web Service"
3. Connect your Git repository
4. Configure the service:
   - **Name**: cdss-api
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt && python -m spacy download en_core_web_sm`
   - **Start Command**: `gunicorn src.api.main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT`
   - **Plan**: Starter (or higher based on your needs)
5. Add the following environment variables:
   - `PYTHON_VERSION`: 3.9.0
   - `ENVIRONMENT`: production
   - `LOG_LEVEL`: INFO
6. Click "Create Web Service"

### Step 2: Deploy the Web Frontend

1. Click "New" and select "Web Service"
2. Connect your Git repository
3. Configure the service:
   - **Name**: cdss-web
   - **Environment**: Docker
   - **Dockerfile Path**: `Dockerfile.web.render`
   - **Plan**: Starter (or higher based on your needs)
4. Add the following environment variables:
   - `API_URL`: (URL of your API service from Step 1)
5. Click "Create Web Service"

## Managing Your Deployment

### Updating Your Application

When you push changes to your repository:

1. Render will automatically rebuild and redeploy your services (if auto-deploy is enabled)
2. You can also manually trigger deployments from the Render Dashboard

### Monitoring

1. Render provides logs for each service
2. You can view logs by clicking on a service and selecting the "Logs" tab
3. Set up alerts in the "Alerts" section of your service

### Scaling

As your application grows:

1. Upgrade your service plan for more resources
2. Enable auto-scaling for services that need it
3. Consider adding a database service for storing prediction logs

## Troubleshooting

### API Service Fails to Start

Check the logs for errors. Common issues include:

- Missing dependencies in requirements.txt
- Model files not properly uploaded
- Environment variables not set correctly

### Web Service Shows Error

Check if the API service is running and accessible. Ensure the `API_URL` environment variable is set correctly.

### Slow Performance

Consider upgrading your service plan or optimizing your code for better performance.

## Cost Optimization

Render's free tier has limitations. To optimize costs:

1. Use the appropriate service plan for your needs
2. Suspend services when not in use during development
3. Optimize your code to reduce resource usage

## Next Steps

After successful deployment:

1. Set up a custom domain for your application
2. Configure SSL for secure communication
3. Implement monitoring and alerting
4. Set up CI/CD for automated testing and deployment
