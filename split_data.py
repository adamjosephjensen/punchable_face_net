import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- Configuration ---
LABELS_FILE = 'data/labels.csv'
OUTPUT_DIR = 'data'
TRAIN_FILE = os.path.join(OUTPUT_DIR, 'train.csv')
DEV_FILE = os.path.join(OUTPUT_DIR, 'dev.csv')
TEST_FILE = os.path.join(OUTPUT_DIR, 'test.csv')

# Split ratios
TEST_SIZE = 0.10 # 10% for the test set
DEV_SIZE = 0.20 # 20% for the dev set (relative to the original dataset)
# Train size will be 1.0 - TEST_SIZE - DEV_SIZE = 70%

# Random state for reproducibility
RANDOM_STATE = 42
# --- End Configuration ---

def split_data():
    """Reads the labels file and splits it into train, dev, and test sets."""

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read the labels
    try:
        df = pd.read_csv(LABELS_FILE)
        print(f"Read {len(df)} labels from {LABELS_FILE}")
    except FileNotFoundError:
        print(f"Error: Labels file not found at {LABELS_FILE}")
        return
    except Exception as e:
        print(f"Error reading {LABELS_FILE}: {e}")
        return

    # Check if dataframe is empty or has required columns
    if df.empty:
        print("Error: Labels file is empty.")
        return
    if 'filename' not in df.columns or 'label' not in df.columns:
        print("Error: Labels file must contain 'filename' and 'label' columns.")
        return

    # Separate features (filenames) and target (labels)
    X = df['filename']
    y = df['label']

    # --- Stratified Splitting ---
    # 1. Split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"Split off {len(X_test)} samples for the test set.")

    # 2. Calculate the proportion of the remaining data needed for the dev set
    # Example: If original was 100, test is 10, remaining is 90.
    # We want dev to be 20% of original (20 samples).
    # So, dev size relative to the remaining data is 20 / 90.
    dev_size_relative = DEV_SIZE / (1.0 - TEST_SIZE)

    # 3. Split the remaining data into train and dev sets
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_temp, y_temp,
        test_size=dev_size_relative,
        random_state=RANDOM_STATE, # Use the same random state for consistency
        stratify=y_temp # Stratify based on the remaining labels
    )
    print(f"Split remaining data into {len(X_train)} train and {len(X_dev)} dev samples.")

    # --- Create Output DataFrames ---
    train_df = pd.DataFrame({'filename': X_train, 'label': y_train})
    dev_df = pd.DataFrame({'filename': X_dev, 'label': y_dev})
    test_df = pd.DataFrame({'filename': X_test, 'label': y_test})

    # --- Save to CSV ---
    try:
        train_df.to_csv(TRAIN_FILE, index=False)
        print(f"Saved training data to {TRAIN_FILE}")
        dev_df.to_csv(DEV_FILE, index=False)
        print(f"Saved development (validation) data to {DEV_FILE}")
        test_df.to_csv(TEST_FILE, index=False)
        print(f"Saved test data to {TEST_FILE}")
    except IOError as e:
        print(f"Error writing split files: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during saving: {e}")

    print("\n--- Split Summary ---")
    print(f"Total samples: {len(df)}")
    print(f"Train samples: {len(train_df)} ({len(train_df)/len(df):.1%})")
    print(f"Dev samples:   {len(dev_df)} ({len(dev_df)/len(df):.1%})")
    print(f"Test samples:  {len(test_df)} ({len(test_df)/len(df):.1%})")
    print("---------------------\n")

if __name__ == "__main__":
    split_data()
