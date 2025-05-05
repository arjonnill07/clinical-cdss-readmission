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
from fastapi.responses import JSONResponse
import uvicorn

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
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
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
            if hasattr(model, 'predict_proba'):
                # For tree-based models
                if hasattr(model, 'estimators_') or 'XGB' in str(type(model)):
                    explainer = shap.TreeExplainer(model)
                else:
                    # For other models
                    explainer = shap.KernelExplainer(
                        model.predict_proba, 
                        shap.sample(X_sample, 100),
                        link="logit"
                    )
            else:
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
    # In a real scenario, you would load the feature list from training
    # and ensure all features are present with appropriate defaults
    
    # For demonstration, we'll just select a subset of features
    selected_features = [
        'age_numeric', 'time_in_hospital', 'num_medications',
        'num_procedures', 'num_diagnoses', 'glucose_level',
        'insulin', 'has_clinical_notes', 'age_med_interaction'
    ]
    
    # Add the los_group dummies
    for col in los_dummies.columns:
        selected_features.append(col)
    
    # Select only features that exist in the DataFrame
    available_features = [f for f in selected_features if f in df.columns]
    
    # Create the feature matrix
    X = df[available_features]
    
    # In a real scenario, you would apply the same scaling as during training
    # For simplicity, we'll skip this step
    
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

@app.get("/")
async def root():
    """Root endpoint."""
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
