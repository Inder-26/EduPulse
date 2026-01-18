from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import sys
import time
from typing import Optional
from datetime import datetime

# --- Constants (Must match Training Config) ---
VOCAB_SIZE = 10000 
MAX_LENGTH = 100 
PADDING_TYPE = 'post'
TRUNC_TYPE = 'post'

# --- Paths ---
MODEL_PATH = os.path.join("artifacts", "model_trainer", "trained_model", "model.keras")
TOKENIZER_PATH = os.path.join("artifacts", "data_transformation", "tokenizer", "tokenizer.pickle")
STATIC_DIR = "static"

app = FastAPI(
    title="EduPulse Sentiment Analysis API",
    description="AI-Powered Educational Feedback Analysis - Analyze sentiment of educational course reviews using Deep Learning.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Resources ---
model = None
tokenizer = None
request_count = 0
start_time = datetime.now()

@app.on_event("startup")
def load_resources():
    global model, tokenizer
    try:
        print("=" * 50)
        print("🚀 EduPulse API Starting...")
        print("=" * 50)
        
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            print(f"✅ Model loaded from {MODEL_PATH}")
        else:
            print(f"❌ ERROR: Model not found at {MODEL_PATH}")

        if os.path.exists(TOKENIZER_PATH):
            with open(TOKENIZER_PATH, 'rb') as handle:
                tokenizer = pickle.load(handle)
            print(f"✅ Tokenizer loaded from {TOKENIZER_PATH}")
        else:
            print(f"❌ ERROR: Tokenizer not found at {TOKENIZER_PATH}")
        
        print("=" * 50)
        print("🎉 EduPulse API Ready!")
        print("=" * 50)
            
    except Exception as e:
        print(f"🔥 CRITICAL ERROR loading resources: {e}")

# Mount static files
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Data Schemas ---
class ReviewRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    sentiment: str
    score: float
    confidence: str
    inference_time_ms: float
    text_length: int
    word_count: int
    star_rating: float

class BatchReviewRequest(BaseModel):
    texts: list[str]

class StatsResponse(BaseModel):
    total_requests: int
    uptime_seconds: float
    model_loaded: bool
    tokenizer_loaded: bool

# --- Endpoints ---
@app.get("/", include_in_schema=False)
def serve_frontend():
    """Serve the main frontend HTML"""
    html_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"message": "Welcome to EduPulse API. Go to /docs for Swagger UI."}

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring"""
    status = "healthy" if model is not None and tokenizer is not None else "degraded"
    return {
        "status": status, 
        "model_loaded": model is not None, 
        "tokenizer_loaded": tokenizer is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/stats", response_model=StatsResponse)
def get_stats():
    """Get API usage statistics"""
    uptime = (datetime.now() - start_time).total_seconds()
    return {
        "total_requests": request_count,
        "uptime_seconds": uptime,
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: ReviewRequest):
    """Analyze sentiment of a single review"""
    global request_count
    request_count += 1
    
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model service not available. Please try again later.")
    
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        start = time.time()
        
        # Preprocess
        text = request.text.strip()
        print(f"DEBUG INPUT: {text}")
        
        sequences = tokenizer.texts_to_sequences([text])
        print(f"DEBUG SEQUENCES: {sequences}")
        
        padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE)
        print(f"DEBUG PADDED: {padded}")
        
        # Predict
        prediction = model.predict(padded, verbose=0)
        print(f"DEBUG PREDICTION: {prediction}")
        
        inference_time = (time.time() - start) * 1000  # Convert to ms
        
        # Logic for 3 classes (Refactored)
        # Class 0: Negative (<= 2.5)
        # Class 1: Neutral (== 3.0)
        # Class 2: Positive (> 3.0)
        
        class_idx = np.argmax(prediction[0])
        score = float(prediction[0][class_idx])
        
        if class_idx == 0:
            sentiment_label = "Negative"
            star_rating = 1.5  # Approximate for display
        elif class_idx == 1:
            sentiment_label = "Neutral"
            star_rating = 3.0
        else:
            sentiment_label = "Positive"
            star_rating = 5.0  # Approximate for display
        
        # Confidence Level
        if score > 0.8:
            confidence_level = "Very High"
        elif score > 0.6:
            confidence_level = "High"
        elif score > 0.4:
            confidence_level = "Moderate"
        else:
            confidence_level = "Low"
        
        return {
            "sentiment": sentiment_label, 
            "score": round(score, 4),
            "confidence": confidence_level,
            "inference_time_ms": round(inference_time, 2),
            "text_length": len(text),
            "word_count": len(text.split()),
            "star_rating": star_rating
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
def predict_batch(request: BatchReviewRequest):
    """Analyze sentiment of multiple reviews"""
    global request_count
    
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model service not available")
    
    if not request.texts or len(request.texts) == 0:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    if len(request.texts) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 texts per batch")
    
    results = []
    for text in request.texts:
        request_count += 1
        try:
            start = time.time()
            sequences = tokenizer.texts_to_sequences([text])
            padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE)
            prediction = model.predict(padded, verbose=0)
            inference_time = (time.time() - start) * 1000
            
            class_idx = np.argmax(prediction[0])
            score = float(prediction[0][class_idx])
            labels = ["Negative", "Positive"]
            label = labels[class_idx]
            
            results.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "sentiment": label,
                "score": round(score, 4),
                "inference_time_ms": round(inference_time, 2)
            })
        except Exception as e:
            results.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "error": str(e)
            })
    
    # Calculate summary
    positive_count = sum(1 for r in results if r.get("sentiment") == "Positive")
    negative_count = sum(1 for r in results if r.get("sentiment") == "Negative")
    
    return {
        "results": results,
        "summary": {
            "total": len(results),
            "positive": positive_count,
            "negative": negative_count,
            "positive_percentage": round(positive_count / len(results) * 100, 1) if results else 0
        }
    }

if __name__ == "__main__":
    import uvicorn
    # For HuggingFace Spaces, use port 7860
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)