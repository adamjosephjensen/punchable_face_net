import os
import argparse
import sys

# --- Configuration ---
# !! IMPORTANT: Verify these paths are correct for your system !!
CELEBA_ATTRIBUTE_FILE = '/Users/adamjensen/Documents/celebA/CelebA/anno/list_attr_celeba.txt'
# Output file will be created in the 'data' subdirectory relative to this script
OUTPUT_DIR = os.path.abspath('./data')
OUTPUT_FILENAME = 'training_imgs.txt'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

TARGET_ATTRIBUTE = "Male"
TARGET_VALUE = "1" # CelebA uses '1' for present, '-1' for absent
DEFAULT_NUM_IMAGES = 20000
# --- End Configuration ---

def find_attribute_column(header_line):
    """Finds the column index for the target attribute."""
    attributes = header_line.split()
    try:
        return attributes.index(TARGET_ATTRIBUTE)
    except ValueError:
        print(f"Error: Target attribute '{TARGET_ATTRIBUTE}' not found in header.")
        print(f"Available attributes: {attributes}")
        sys.exit(1)

def generate_list(num_images):
    """Reads the attribute file and generates the list of filenames."""
    print(f"Reading attributes from: {CELEBA_ATTRIBUTE_FILE}")
    print(f"Looking for attribute '{TARGET_ATTRIBUTE}' with value '{TARGET_VALUE}'.")
    print(f"Target number of images: {num_images}")

    if not os.path.exists(CELEBA_ATTRIBUTE_FILE):
        print(f"Error: Attribute file not found at '{CELEBA_ATTRIBUTE_FILE}'")
        sys.exit(1)

    selected_filenames = []
    try:
        with open(CELEBA_ATTRIBUTE_FILE, 'r') as f:
            # Read total number of images (line 1) - not strictly needed here
            f.readline()
            # Read header line (line 2)
            header = f.readline().strip()
            target_column_index = find_attribute_column(header)
            print(f"Found '{TARGET_ATTRIBUTE}' at column index {target_column_index}.")

            # Read image attribute lines
            line_num = 2 # Start counting after headers
            for line in f:
                line_num += 1
                parts = line.split()
                if not parts: # Skip empty lines if any
                    continue

                filename = parts[0]
                attributes = parts[1:] # Attributes start from the second element

                if len(attributes) <= target_column_index:
                     print(f"Warning: Line {line_num} ('{filename}') has fewer columns than expected. Skipping.")
                     continue

                # Check if the target attribute matches the target value
                if attributes[target_column_index] == TARGET_VALUE:
                    selected_filenames.append(filename)

                # Stop if we have enough filenames
                if len(selected_filenames) >= num_images:
                    break

    except IOError as e:
        print(f"Error reading attribute file: {e}")
        sys.exit(1)
    except Exception as e:
         print(f"An unexpected error occurred during file processing: {e}")
         sys.exit(1)


    if len(selected_filenames) < num_images:
        print(f"\nWarning: Found only {len(selected_filenames)} images matching the criteria, "
              f"which is less than the requested {num_images}.")
    else:
         print(f"\nFound {len(selected_filenames)} matching images.")

    # Ensure output directory exists
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Ensured output directory exists: {OUTPUT_DIR}")
    except OSError as e:
        print(f"Error creating output directory '{OUTPUT_DIR}': {e}")
        sys.exit(1)


    # Write the selected filenames to the output file
    try:
        with open(OUTPUT_FILE, 'w') as outfile:
            for fname in selected_filenames:
                outfile.write(fname + '\n')
        print(f"Successfully wrote {len(selected_filenames)} filenames to: {OUTPUT_FILE}")
    except IOError as e:
        print(f"Error writing to output file '{OUTPUT_FILE}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Generate a list of CelebA image filenames based on the '{TARGET_ATTRIBUTE}' attribute.")
    parser.add_argument(
        '-n', '--num_images',
        type=int,
        default=DEFAULT_NUM_IMAGES,
        help=f"Number of images to select (default: {DEFAULT_NUM_IMAGES})"
    )
    args = parser.parse_args()

    if args.num_images <= 0:
         print("Error: Number of images must be positive.")
         sys.exit(1)

    generate_list(args.num_images)
