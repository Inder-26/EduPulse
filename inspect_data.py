
import numpy as np
import pandas as pd

try:
    # Load transformed data
    train_arr = np.load("artifacts/data_transformation/transformed/train.npy", allow_pickle=True)
    
    # Extract labels (index 1)
    y_train = train_arr[1]
    
    # Check if one-hot or simple labels
    if y_train.ndim > 1:
        # Convert back from one-hot if needed (though it's usually saved as labels before one-hot in this pipeline)
        y_indices = np.argmax(y_train, axis=1)
    else:
        y_indices = y_train
        
    # Count classes
    unique, counts = np.unique(y_indices, return_counts=True)
    total = sum(counts)
    
    print("\n--- Class Distribution ---")
    print(f"Total Samples: {total}")
    for cls, count in zip(unique, counts):
        percentage = (count / total) * 100
        print(f"Class {cls}: {count} samples ({percentage:.2f}%)")
        
    print("\nNOTE: Class 0 corresponds to Rating 1.0, Class 8 to Rating 5.0")

except Exception as e:
    print(f"Error: {e}")
