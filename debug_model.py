
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Constants
MODEL_PATH = os.path.join("artifacts", "model_trainer", "trained_model", "model.keras")
TOKENIZER_PATH = os.path.join("artifacts", "data_transformation", "tokenizer", "tokenizer.pickle")
MAX_LENGTH = 100
PADDING_TYPE = 'post'
TRUNC_TYPE = 'post'

def debug_prediction(text):
    print(f"\n--- Debugging: '{text}' ---")
    
    # Load Resources
    try:
        model = load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as handle:
            tokenizer = pickle.load(handle)
    except Exception as e:
        print(f"Error loading resources: {e}")
        return

    # 1. Tokenization
    sequences = tokenizer.texts_to_sequences([text])
    print(f"Raw Tokens: {sequences}")
    
    if not sequences or not sequences[0]:
        print("CRITICAL: Text resulted in empty sequence! Words not in vocabulary?")
        # Check individual words
        for word in text.split():
            idx = tokenizer.word_index.get(word.lower())
            print(f"Word: '{word}' -> Index: {idx}")
        return

    # 2. Padding
    padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE)
    print(f"Padded Sequence: {padded}")

    # 3. Prediction
    prediction = model.predict(padded, verbose=0)
    print(f"Raw Prediction (Softmax): {prediction[0]}")
    
    # 4. Interpretation
    class_idx = np.argmax(prediction[0])
    score = float(prediction[0][class_idx])
    star_rating = 1.0 + (class_idx * 0.5)
    
    print(f"Predicted Class Index: {class_idx}")
    print(f"Predicted Star Rating: {star_rating}")
    print(f"Confidence: {score:.4f}")

if __name__ == "__main__":
    debug_prediction("very bad course")
    debug_prediction("excellent course")
