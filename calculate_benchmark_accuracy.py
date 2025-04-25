import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report
import os
import sys

# --- Configuration ---
DEV_ORIGINAL_FILE = os.path.abspath('./data/dev.csv')
DEV_RERATED_FILE = os.path.abspath('./data/dev_rerated.csv')
# --- End Configuration ---

def calculate_benchmark():
    """
    Calculates and prints human benchmark performance by comparing original
    dev set labels with re-rated labels. Focuses on accuracy and
    Quadratic Weighted Kappa for ordinal agreement.
    """
    print("--- Calculating Human Benchmark Performance (Self-Consistency) ---")

    # --- Input File Validation ---
    if not os.path.exists(DEV_ORIGINAL_FILE):
        print(f"Error: Original dev file not found at {DEV_ORIGINAL_FILE}")
        sys.exit(1)
    if not os.path.exists(DEV_RERATED_FILE):
        print(f"Error: Re-rated dev file not found at {DEV_RERATED_FILE}")
        print("Please run the labeler in re-rate mode first (`python app/labeler.py --mode rerate`).")
        sys.exit(1)

    try:
        # --- Load Data ---
        print(f"Loading original labels from: {DEV_ORIGINAL_FILE}")
        df_original = pd.read_csv(DEV_ORIGINAL_FILE)
        print(f"Loading re-rated labels from: {DEV_RERATED_FILE}")
        df_rerated = pd.read_csv(DEV_RERATED_FILE)

        # --- Data Validation ---
        if df_original.empty:
            print(f"Error: Original dev file '{DEV_ORIGINAL_FILE}' is empty.")
            sys.exit(1)
        if df_rerated.empty:
             print(f"Error: Re-rated dev file '{DEV_RERATED_FILE}' is empty.")
             sys.exit(1)

        required_cols_orig = ['filename', 'label']
        required_cols_rerated = ['filename', 'new_label']
        if not all(col in df_original.columns for col in required_cols_orig):
             print(f"Error: {DEV_ORIGINAL_FILE} missing required columns. Expected: {required_cols_orig}")
             sys.exit(1)
        if not all(col in df_rerated.columns for col in required_cols_rerated):
             print(f"Error: {DEV_RERATED_FILE} missing required columns. Expected: {required_cols_rerated}")
             sys.exit(1)

        # --- Merge Data ---
        # Use inner merge to only compare images present in both files
        print("Merging original and re-rated data on 'filename'...")
        df_merged = pd.merge(
            df_original[['filename', 'label']],
            df_rerated[['filename', 'new_label']],
            on='filename',
            how='inner' # Important: only compare images rated both times
        )

        if df_merged.empty:
            print("Error: No matching filenames found between original and re-rated dev sets.")
            print(f"Ensure filenames in '{os.path.basename(DEV_RERATED_FILE)}' correspond to those in '{os.path.basename(DEV_ORIGINAL_FILE)}'.")
            sys.exit(1)

        print(f"Found {len(df_merged)} images common to both files for comparison.")

        # --- Extract Labels ---
        original_labels = df_merged['label']
        new_labels = df_merged['new_label']

        # --- Calculate Metrics ---
        print("Calculating metrics...")
        accuracy = accuracy_score(original_labels, new_labels)
        # Use quadratic weights for Cohen's Kappa, suitable for ordinal data
        kappa = cohen_kappa_score(original_labels, new_labels, weights='quadratic')

        # --- Print Results ---
        print("\n--- Benchmark Results ---")
        print(f"Images Compared: {len(df_merged)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Quadratic Weighted Kappa: {kappa:.4f}")
        print("\nClassification Report (Original vs. Re-rated):")
        # Provides precision, recall, f1-score per class. Useful for seeing class-specific agreement.
        # Use zero_division=0 to avoid warnings if a class has no support in either set.
        print(classification_report(original_labels, new_labels, zero_division=0))
        print("-------------------------\n")
        print("Interpretation:")
        print(" - Accuracy: Percentage of times you assigned the exact same label.")
        print(" - Quadratic Weighted Kappa: Measures agreement, penalizing large disagreements more than small ones (scale -1 to 1).")
        print("   Values > 0.8 indicate almost perfect agreement.")
        print("   Values 0.6-0.8 indicate substantial agreement.")
        print("   Values 0.4-0.6 indicate moderate agreement.")
        print("   Lower values suggest inconsistency or difficulty in applying the criteria.")
        print("This score serves as a practical upper bound (Human Level Performance) for your model on this task.")


    except pd.errors.EmptyDataError:
        print(f"Error: One or both CSV files could not be read properly or are empty.")
        sys.exit(1)
    except FileNotFoundError:
         # Should be caught earlier, but as a fallback
         print("Error: One of the required CSV files was not found.")
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during calculation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    calculate_benchmark()
