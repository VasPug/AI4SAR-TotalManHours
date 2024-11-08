import pickle
import re
from pathlib import Path

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent


with open(f"{BASE_DIR}/trained_pipeline-{__version__}.pkl", "rb") as f:
    model = pickle.load(f)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# Generalization Mapping
subject_category_mapping = {
    'Child': 'Minor', 'Youth': 'Minor',
    'Dementia': 'Medical', 'Mental Illness': 'Medical', 'Autistic': 'Medical', 'Intellectual Disability': 'Medical',
    'Substance Intoxication': 'Substance', 'Drinking': 'Substance', 'drugs': 'Substance',
    'Hiker': 'Outdoor Activity', 'Hunter': 'Outdoor Activity', 'Climber': 'Outdoor Activity', 'Angler': 'Outdoor Activity',
    'Snowboarder': 'Outdoor Activity', 'Runner': 'Outdoor Activity', 'Caver': 'Outdoor Activity', 'Gatherer': 'Outdoor Activity',
    'Skier-Nordic': 'Outdoor Activity', 'Skier-Alpine': 'Outdoor Activity', 'ATV': 'Outdoor Activity', 'Camper': 'Outdoor Activity', 'Horseback Rider': 'Outdoor Activity',
    'Aircraft': 'Transport', 'Vehicle': 'Transport', 'Vehicle-4wd': 'Transport', 'Motorcycle': 'Transport', 'Aircraft-nonpowered': 'Transport',
    'Worker': 'Occupational', 'Boater': 'Water Activity', 'Boating': 'Water Activity', 'Fishing': 'Water Activity',
    'Despondent': 'Emotional', 'Abduction': 'Emotional', 'Runaway': 'Emotional', 'Walkaway': 'Emotional', 'playing': 'Emotional', 'outing': 'Emotional',
    'Playing': 'Leisure', 'Rafting': 'Leisure', 'Mushroom Hunting': 'Leisure', 'Gathering': 'Leisure', 'Camping': 'Leisure', 'School': 'Leisure',
    'Extreme Sports': 'Extreme Activity', 'Criminal': 'Illegal Activity', 'Evading': 'Illegal Activity'
}

subject_activity_mapping = {
    'Playing': 'Leisure', 'playing': 'Leisure', 'Runaway': 'Leisure', 'Walkaway': 'Leisure', 'Tramping': 'Leisure', 'hiking': 'Leisure', 'Hiking': 'Leisure', 'Camping': 'Leisure', 'outing': 'Leisure', 'Swimming': 'Leisure',
    'Drinking': 'Substance', 'drugs': 'Substance',
    'Hunting': 'Outdoor Activity', 'hunting': 'Outdoor Activity', 'Fishing': 'Outdoor Activity', 'fishing': 'Outdoor Activity', 'Climbing': 'Outdoor Activity', 'Biking': 'Outdoor Activity', 'Rafting': 'Outdoor Activity', 'Caving': 'Outdoor Activity', 'Mushroom Hunting': 'Outdoor Activity', 'Gathering': 'Outdoor Activity', 'Riding': 'Outdoor Activity', 'Skiing': 'Outdoor Activity', 'Snowboarding': 'Outdoor Activity',
    'Flying': 'Transport', 'Driving': 'Transport',
    'Working': 'Occupational',
    'Boating': 'Water Activity',
    'Extreme Sports': 'Extreme Activity',
    'Criminal': 'Illegal Activity', 'Evading': 'Illegal Activity', 'hiding': 'Illegal Activity',
    'School': 'Education'
}

weather_mapping = {
    'Clear': 'Clear', 'Sunny': 'Clear',
    'Partly cloudy': 'Cloudy', 'Partly Cloudy': 'Cloudy', 'Cloudy': 'Cloudy', 'Overcast': 'Cloudy',
    'Foggy': 'Foggy',
    'Rain': 'Rain', 'Drizzle': 'Rain', 'Showers': 'Rain', 'Thunderstorm': 'Rain',
    'Snow': 'Snow', 'Sleet': 'Snow', 'Hail': 'Snow'
}

categorical_features = [ 'Subject.Category', 'Weather', 'Subject.Activity']
numerical_features = ['Age']

def predict_pipeline(subject_category, subject_activity, weather, age):
    # Apply the generalization mappings
    generalized_subject_category = subject_category_mapping.get(subject_category, subject_category)
    generalized_subject_activity = subject_activity_mapping.get(subject_activity, subject_activity)
    generalized_weather = weather_mapping.get(weather, weather)

    # Prepare input data as a DataFrame to match the training format
    X_new = pd.DataFrame({
        'Subject.Category': [generalized_subject_category],
        'Subject.Activity': [generalized_subject_activity],
        'Weather': [generalized_weather],
        'Age': [age]
    })

    # Ensure the DataFrame has the same categorical and numerical features as the training data
    X_new = X_new[categorical_features + numerical_features]

    # Make prediction using the model
    # pred = model.predict(X_new)

    return pred
