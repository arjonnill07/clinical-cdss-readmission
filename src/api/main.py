"""
FastAPI application for the Clinical CDSS Readmission prediction service.

This script:
1. Loads the trained model
2. Defines API endpoints for prediction
3. Implements input validation
4. Provides model explanations
5. Logs predictions for monitoring

Theory:
- RESTful APIs provide a standard interface for model serving
- Input validation ensures data quality
- Explanations help clinicians understand predictions
- Logging enables monitoring and auditing
- Versioning supports model updates
"""

import os
import sys
import logging
from pathlib import Path
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field, validator
import shap
from fastapi import FastAPI, HTTPException, Depends, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
MODELS_DIR = Path('models')
FEATURES_DATA_DIR = Path('data/features')

# Create FastAPI app
app = FastAPI(
    title="Clinical CDSS Readmission Prediction API",
    description="API for predicting 30-day hospital readmissions for diabetic patients",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Input models
class PatientFeatures(BaseModel):
    """
    Model for patient features input.

    This is a simplified example. In a real-world scenario,
    you would have more specific fields based on your data.
    """
    age: int = Field(..., description="Patient age", ge=0, le=120)
    gender: str = Field(..., description="Patient gender", example="Male")
    time_in_hospital: int = Field(..., description="Length of stay in days", ge=0)
    num_medications: int = Field(..., description="Number of medications", ge=0)
    num_procedures: int = Field(..., description="Number of procedures", ge=0)
    num_diagnoses: int = Field(..., description="Number of diagnoses", ge=0)
    glucose_level: float = Field(..., description="Blood glucose level")
    A1C_level: Optional[float] = Field(None, description="HbA1c level")
    insulin: bool = Field(..., description="Whether insulin was prescribed")
    clinical_notes: Optional[str] = Field(None, description="Clinical notes text")

    # Additional fields can be added based on your model's requirements

    @validator('gender')
    def validate_gender(cls, v):
        allowed_values = ['Male', 'Female', 'Other']
        if v not in allowed_values:
            raise ValueError(f"Gender must be one of {allowed_values}")
        return v

class PredictionResponse(BaseModel):
    """Model for prediction response."""
    patient_id: str
    readmission_risk: float
    readmission_prediction: bool
    risk_level: str
    timestamp: str
    model_version: str
    explanation: Dict[str, float]

# Global variables to store model and related objects
model = None
model_info = None
feature_names = None
explainer = None

@app.on_event("startup")
async def startup_event():
    """Load model and related objects on startup."""
    global model, model_info, feature_names, explainer

    try:
        # Load model
        logger.info("Loading model")
        model_path = MODELS_DIR / 'model.pkl'
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Load model info
        info_path = MODELS_DIR / 'model_info.json'
        with open(info_path, 'r') as f:
            model_info = json.load(f)

        feature_names = model_info.get('feature_names', [])

        # Load SHAP explainer
        try:
            logger.info("Loading SHAP explainer")
            with open(MODELS_DIR / 'shap_values.pkl', 'rb') as f:
                shap_data = pickle.load(f)

            # Create a sample of the data for the explainer
            X_sample = shap_data['X_sample']

            # Create SHAP explainer based on model type
            model_type = str(type(model))
            logger.info(f"Creating SHAP explainer for model type: {model_type}")

            if hasattr(model, 'predict_proba'):
                # For tree-based models (Random Forest, Gradient Boosting, XGBoost)
                if ('GradientBoostingClassifier' in model_type or
                    'RandomForestClassifier' in model_type or
                    'XGBClassifier' in model_type or
                    hasattr(model, 'estimators_')):
                    logger.info("Using TreeExplainer for tree-based model")
                    explainer = shap.TreeExplainer(model)
                else:
                    # For other models
                    logger.info("Using KernelExplainer with predict_proba")
                    explainer = shap.KernelExplainer(
                        model.predict_proba,
                        shap.sample(X_sample, 100),
                        link="logit"
                    )
            else:
                logger.info("Using KernelExplainer with predict")
                explainer = shap.KernelExplainer(model.predict, shap.sample(X_sample, 100))

            logger.info("SHAP explainer loaded successfully")
        except Exception as e:
            logger.warning(f"Error loading SHAP explainer: {e}")
            explainer = None

        logger.info("Model and related objects loaded successfully")

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def preprocess_input(patient: PatientFeatures) -> pd.DataFrame:
    """
    Preprocess the input data to match the model's expected format.

    Args:
        patient: Patient features from the API request

    Returns:
        DataFrame with preprocessed features
    """
    # Convert patient data to dictionary
    patient_dict = patient.dict()

    # Create a DataFrame with a single row
    df = pd.DataFrame([patient_dict])

    # Perform the same preprocessing steps as during training

    # 1. Create derived features
    df['age_numeric'] = df['age']

    # 1.1 Encode categorical variables
    if 'gender' in df.columns:
        # Map gender to numeric values (similar to what was done in training)
        gender_mapping = {'Male': 0, 'Female': 1, 'Other': 2}
        df['gender'] = df['gender'].map(gender_mapping).fillna(0).astype(int)
        logger.info(f"Encoded gender: {df['gender'].iloc[0]}")

    # 2. Handle clinical notes if present
    if 'clinical_notes' in df.columns and df['clinical_notes'].iloc[0]:
        # In a real scenario, you would apply the same NLP processing
        # For simplicity, we'll just create a dummy feature
        df['has_clinical_notes'] = 1
    else:
        df['has_clinical_notes'] = 0

    # 3. Create interaction features
    df['age_med_interaction'] = df['age'] * df['num_medications']

    # 4. Create time-based features
    df['los_group'] = pd.cut(
        df['time_in_hospital'],
        bins=[0, 3, 7, 14, float('inf')],
        labels=['short', 'medium', 'long', 'very_long']
    ).astype(str)

    # One-hot encode the binned feature
    los_dummies = pd.get_dummies(df['los_group'], prefix='los')
    df = pd.concat([df, los_dummies], axis=1)

    # 5. Ensure all required features are present
    # Load the feature list from the trained model
    try:
        # Try to get feature names from model info
        if feature_names and len(feature_names) > 0:
            selected_features = feature_names
            logger.info(f"Using {len(selected_features)} features from model info")
        else:
            # Fall back to a subset of features if model info is not available
            selected_features = [
                'age_numeric', 'time_in_hospital', 'num_medications',
                'num_procedures', 'num_diagnoses', 'glucose_level',
                'insulin', 'has_clinical_notes', 'age_med_interaction'
            ]

            # Add the los_group dummies
            for col in los_dummies.columns:
                selected_features.append(col)

            logger.warning("Using fallback feature list as model feature names not available")
    except Exception as e:
        # Fall back to a subset of features if there's an error
        selected_features = [
            'age_numeric', 'time_in_hospital', 'num_medications',
            'num_procedures', 'num_diagnoses', 'glucose_level',
            'insulin', 'has_clinical_notes', 'age_med_interaction'
        ]

        # Add the los_group dummies
        for col in los_dummies.columns:
            selected_features.append(col)

        logger.warning(f"Error getting model features: {e}. Using fallback feature list.")

    # Select only features that exist in the DataFrame
    available_features = [f for f in selected_features if f in df.columns]

    # Check for missing features and add them with default values
    missing_features = [f for f in selected_features if f not in df.columns]
    if missing_features:
        logger.warning(f"Missing features: {missing_features}. Adding with default values of 0.")
        for feature in missing_features:
            df[feature] = 0
        available_features = selected_features

    # Create the feature matrix
    X = df[available_features]

    # Apply the same scaling as during training
    try:
        # Load the scaler
        with open(FEATURES_DATA_DIR / 'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # Get numerical columns (excluding any categorical columns that might have been one-hot encoded)
        num_cols = X.select_dtypes(include=['number']).columns

        # Apply scaling
        X[num_cols] = scaler.transform(X[num_cols])
        logger.info("Applied feature scaling to input data")
    except Exception as e:
        logger.warning(f"Could not apply scaling: {e}. Proceeding with unscaled features.")

    return X

def get_explanation(X: pd.DataFrame) -> Dict[str, float]:
    """
    Generate explanation for the prediction.

    Args:
        X: Preprocessed features

    Returns:
        Dictionary mapping feature names to their importance values
    """
    if explainer is None:
        logger.warning("SHAP explainer not available. Returning empty explanation.")
        return {}

    try:
        # Calculate SHAP values
        shap_values = explainer.shap_values(X)

        # If shap_values is a list (for multi-class), take the values for class 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Create a dictionary mapping feature names to their importance
        feature_importance = {}
        for i, col in enumerate(X.columns):
            feature_importance[col] = float(shap_values[0][i])

        # Sort by absolute importance
        sorted_importance = {
            k: v for k, v in sorted(
                feature_importance.items(),
                key=lambda item: abs(item[1]),
                reverse=True
            )
        }

        # Return top 10 features
        return dict(list(sorted_importance.items())[:10])

    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        return {}

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint that serves the frontend."""
    try:
        # Path to the index.html file
        frontend_path = Path('frontend/index.html')

        if frontend_path.exists():
            with open(frontend_path, 'r') as f:
                html_content = f.read()
            return HTMLResponse(content=html_content)
        else:
            # Fallback if the file doesn't exist
            return HTMLResponse(content="""
            <!DOCTYPE html>
            <html>
                <head>
                    <title>Clinical CDSS Readmission Prediction</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }
                        .container { max-width: 800px; margin: 0 auto; }
                        h1 { color: #0066cc; }
                        .card { border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
                        .btn { display: inline-block; background: #0066cc; color: white; padding: 10px 15px; text-decoration: none; border-radius: 4px; }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>Clinical CDSS Readmission Prediction</h1>
                        <div class="card">
                            <h2>Welcome to the API</h2>
                            <p>This is the API for the Clinical Decision Support System for predicting patient readmissions.</p>
                            <p>The frontend is currently being built. You can access the API documentation at <a href="/docs">/docs</a>.</p>
                            <a href="/docs" class="btn">API Documentation</a>
                        </div>
                    </div>
                </body>
            </html>
            """)
    except Exception as e:
        logger.error(f"Error serving frontend: {e}")
        return {
            "message": "Clinical CDSS Readmission Prediction API",
            "version": "1.0.0",
            "status": "active"
        }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    return {"status": "healthy", "model_version": model_info.get("model_name", "unknown")}

@app.get("/model/info")
async def get_model_info():
    """Get information about the model."""
    if model_info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model information not available"
        )
    return model_info

@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientFeatures):
    """
    Predict readmission risk for a patient.

    Args:
        patient: Patient features

    Returns:
        Prediction response with risk score and explanation
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        # Generate a patient ID (in a real system, this would be provided)
        patient_id = f"P{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Preprocess input
        X = preprocess_input(patient)

        # Make prediction
        risk_score = float(model.predict_proba(X)[0, 1])
        prediction = bool(risk_score >= 0.5)

        # Determine risk level
        if risk_score < 0.3:
            risk_level = "Low"
        elif risk_score < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # Generate explanation
        explanation = get_explanation(X)

        # Create response
        response = PredictionResponse(
            patient_id=patient_id,
            readmission_risk=risk_score,
            readmission_prediction=prediction,
            risk_level=risk_level,
            timestamp=datetime.now().isoformat(),
            model_version=model_info.get("model_name", "unknown"),
            explanation=explanation
        )

        # Log prediction (in a real system, you would store this in a database)
        logger.info(f"Prediction for {patient_id}: {risk_score:.4f} ({risk_level})")

        return response

    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making prediction: {str(e)}"
        )

@app.get("/features")
async def get_features():
    """Get the list of features used by the model."""
    if feature_names is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feature information not available"
        )
    return {"features": feature_names}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
