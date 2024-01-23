import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import re
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.StreamHandler(),
        # Add more handlers if needed
    ]
)

# Load the trained model
clf = joblib.load('/Users/dikshitrishi/Desktop/Codes/News-Article-Sorting/model/article_classifier_model.joblib')

# FastAPI instance
app = FastAPI()

# Pydantic model for input validation
class TextRequest(BaseModel):
    text: str

# Prediction route
@app.post("/predict")
def predict_category(text_request: TextRequest):
    try:
        text = text_request.text

        # Text Preprocessing
        def preprocess(text):
            text = re.sub(r'[^\w\s\']', ' ', text)
            text = re.sub(r'[ \n]+', ' ', text)
            return text.strip().lower()

        # Converting into array format
        def input(a):
            arr = np.array([a])
            return arr

        # Measure latency
        start_time = time.time()

        preprocessed_text = preprocess(text)
        corr_form = input(preprocessed_text)
        prediction = clf.predict(corr_form)

        end_time = time.time()
        latency = end_time - start_time

        # Mapping back to category names
        categories = {0: 'business', 1: 'sports', 2: 'politics', 3: 'entertainment', 4: 'tech'}

        # Get the category or default message
        category_message = categories.get(prediction[0], 'Not covered in category of business, sports, politics, entertainment and tech.')

        # Log the result along with latency
        logging.info(f'It is a {category_message} news. Latency: {latency} seconds')

        # Return the result as a JSON response
        return {"prediction": category_message, "latency": latency}
    except Exception as e:
        # Log the error details
        logging.error(f"Error: {str(e)}")
        # Raise an HTTPException with a 500 status code
        raise HTTPException(status_code=500, detail="Internal Server Error")
