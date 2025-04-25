import os
import csv
import random
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory

# --- Configuration ---
# IMAGE_DIR = os.path.abspath('./data/images') # No longer copying images locally
# !! IMPORTANT: Verify this path is correct for your system !!
CELEBA_IMAGE_DIR = '/Users/adamjensen/Documents/celebA/CelebA/Img/img_align_celeba'
TRAINING_LIST_FILE = os.path.abspath('./data/training_imgs.txt')
LABELS_FILE = os.path.abspath('./data/labels.csv')
SKIPPED_FILE = os.path.abspath('./data/skipped.txt') # New
FLAGGED_FILE = os.path.abspath('./data/flagged.txt') # New
DEV_FILE = os.path.abspath('./data/dev.csv') # Need path to dev set
DEV_RERATED_FILE = os.path.abspath('./data/dev_rerated.csv') # New file for re-ratings
# IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'} # No longer needed when reading from list
BATCH_SIZE = 10 # How many images to label before showing progress

# --- Label Mapping ---
LABEL_MAP = {
    "very_punchable": 3,
    "punchable": 2,
    "not_punchable": 1,
    "very_not_punchable": 0
}
# Define the set of valid action strings (labels + skip + flag)
VALID_ACTIONS = set(LABEL_MAP.keys()) | {'skip', 'flag'}

# --- Flask App Setup ---
# Calculate paths relative to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.abspath(os.path.join(script_dir, '..', 'templates'))
static_dir = os.path.abspath(os.path.join(script_dir, '..', 'data'))

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.secret_key = 'super secret key' # Change this for production use

# --- Global State for Undo ---
# Stores the details of the last action for potential undo
# Example: {'action': 'label', 'filename': '000001.jpg', 'value': 3}
# Example: {'action': 'skip', 'filename': '000002.jpg', 'value': None}
# Example: {'action': 'flag', 'filename': '000003.jpg', 'value': None}
last_action_info = None

# --- Helper Functions ---
def get_all_images():
    """Gets a list of image filenames from the training list file."""
    images = []
    if not os.path.exists(TRAINING_LIST_FILE):
        print(f"Error: Training list file not found at {TRAINING_LIST_FILE}")
        print("Please run the generate_training_list.py script first.")
        # Return empty list, Flask route will show an appropriate message
        return images
    try:
        with open(TRAINING_LIST_FILE, 'r') as f:
            images = [line.strip() for line in f if line.strip()]
    except IOError as e:
        print(f"Error reading training list file {TRAINING_LIST_FILE}: {e}")
    return images

def get_processed_images():
    """Gets sets of filenames that have been labeled, skipped, or flagged."""
    labeled = set()
    if os.path.exists(LABELS_FILE):
        try:
            with open(LABELS_FILE, 'r', newline='') as f:
                reader = csv.reader(f)
                try:
                    header = next(reader) # Skip header
                    if header != ['filename', 'label']:
                         print(f"Warning: Unexpected header in {LABELS_FILE}. Re-writing.")
                         write_labels_header() # Attempt to fix header
                    else:
                        for row in reader:
                            if row and len(row) >= 1: # Check row exists and has at least one element
                                labeled.add(row[0])
                except StopIteration:
                     # File is empty or only contains header
                     if not os.path.exists(LABELS_FILE) or os.path.getsize(LABELS_FILE) == 0:
                          write_labels_header() # Ensure header exists if file is empty
        except (IOError, csv.Error, IndexError) as e:
            print(f"Warning: Could not read or parse labels file {LABELS_FILE}. Treating as empty. Error: {e}")
            write_labels_header() # Ensure header exists

    skipped = read_set_from_file(SKIPPED_FILE)
    flagged = read_set_from_file(FLAGGED_FILE)

    return labeled, skipped, flagged

# --- File I/O Helpers ---

def read_set_from_file(filepath):
    """Reads lines from a file into a set."""
    items = set()
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                items = {line.strip() for line in f if line.strip()}
        except IOError as e:
            print(f"Warning: Could not read file {filepath}. Error: {e}")
    return items

def append_line_to_file(filepath, line):
    """Appends a line to a file."""
    try:
        with open(filepath, 'a') as f:
            f.write(line + '\n')
    except IOError as e:
        print(f"Error: Could not append to file {filepath}. Error: {e}")
        flash(f"Error saving state to {os.path.basename(filepath)}", "error") # User feedback

def remove_last_line_matching(filepath, filename_to_match, is_csv=False):
    """
    Reads a file, removes the *last* line matching the filename, and rewrites the file.
    For CSV, matches the first column. For txt, matches the whole line.
    Returns True if a line was removed, False otherwise.
    """
    lines_to_keep = []
    found_match = False
    last_match_index = -1

    if not os.path.exists(filepath):
        return False # Nothing to remove

    try:
        with open(filepath, 'r', newline='' if is_csv else None) as f:
            if is_csv:
                reader = csv.reader(f)
                all_lines = list(reader)
                if not all_lines: return False # Empty file
                header = all_lines[0]
                data_lines = all_lines[1:]
                # Find the index of the last row matching the filename
                for i in range(len(data_lines) - 1, -1, -1):
                    if data_lines[i] and data_lines[i][0] == filename_to_match:
                        last_match_index = i
                        found_match = True
                        break
                if found_match:
                    lines_to_keep = [header] + data_lines[:last_match_index] + data_lines[last_match_index+1:]
                else:
                    return False # No match found
            else: # Simple text file
                all_lines = f.readlines()
                # Find the index of the last line matching the filename
                for i in range(len(all_lines) - 1, -1, -1):
                    if all_lines[i].strip() == filename_to_match:
                        last_match_index = i
                        found_match = True
                        break
                if found_match:
                    lines_to_keep = all_lines[:last_match_index] + all_lines[last_match_index+1:]
                else:
                    return False # No match found

        # Rewrite the file with the matching line removed
        with open(filepath, 'w', newline='' if is_csv else None) as f:
            if is_csv:
                writer = csv.writer(f)
                writer.writerows(lines_to_keep)
            else:
                f.writelines(lines_to_keep)
        return True

    except (IOError, csv.Error, IndexError) as e:
        print(f"Error processing file {filepath} for removal: {e}")
        flash(f"Error updating {os.path.basename(filepath)} during undo.", "error")
        return False


def write_labels_header():
    """Writes the header row to the CSV file if it doesn't exist or is empty."""
    # Check if file exists and has content beyond just a header maybe
    write_header = True
    if os.path.exists(LABELS_FILE) and os.path.getsize(LABELS_FILE) > 10: # Arbitrary small size check
         try:
             with open(LABELS_FILE, 'r', newline='') as f:
                 reader = csv.reader(f)
                 header = next(reader)
                 if header == ['filename', 'label']:
                     write_header = False # Header exists
         except (IOError, StopIteration, csv.Error):
             pass # Problem reading, better to rewrite header

    if write_header:
        try:
            with open(LABELS_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                # Header remains the same, but meaning changes: 0-3 scale
                writer.writerow(['filename', 'label']) # label: 3=very_punchable, 2=punchable, 1=not_punchable, 0=very_not_punchable
        except IOError as e:
            print(f"Error: Could not write header to labels file {LABELS_FILE}. Error: {e}")


def get_next_image_to_label(all_images, labeled_images, skipped_images, flagged_images):
    """
    Finds the next image to label according to priority:
    1. Unseen (not labeled, not skipped, not flagged)
    2. Skipped (skipped, but not labeled or flagged)
    """
    processed_images = labeled_images | skipped_images | flagged_images
    unseen = [img for img in all_images if img not in processed_images]

    if unseen:
        print(f"Selecting from {len(unseen)} unseen images.")
        return random.choice(unseen)

    # If no unseen images, check for skippable images
    skippable = [img for img in skipped_images if img not in labeled_images and img not in flagged_images]
    if skippable:
        print(f"No unseen images left. Selecting from {len(skippable)} skipped images.")
        return random.choice(skippable)

    # No unseen or skippable images left
    print("No more images to label (unseen or skipped).")
    return None

# --- Re-rating Mode Helper Functions ---

def get_dev_images_to_rerate():
    """Reads the dev set CSV and returns a list of filenames."""
    dev_images = []
    if not os.path.exists(DEV_FILE):
        print(f"Error: Development set file not found at {DEV_FILE}")
        flash(f"Error: Dev set file '{os.path.basename(DEV_FILE)}' not found. Cannot start re-rating.", "error")
        return dev_images # Return empty list

    try:
        # Read only the 'filename' column
        df_dev = pd.read_csv(DEV_FILE, usecols=['filename'])
        dev_images = df_dev['filename'].tolist()
    except (IOError, pd.errors.EmptyDataError, ValueError, KeyError) as e:
        print(f"Error reading or parsing dev file {DEV_FILE}: {e}")
        flash(f"Error reading dev set file '{os.path.basename(DEV_FILE)}'.", "error")
    return dev_images

def get_rerated_images():
    """Reads the dev_rerated.csv file and returns a set of already re-rated filenames."""
    rerated = set()
    if os.path.exists(DEV_RERATED_FILE):
        try:
            with open(DEV_RERATED_FILE, 'r', newline='') as f:
                reader = csv.reader(f)
                try:
                    header = next(reader) # Skip header
                    if header != ['filename', 'new_label']:
                         print(f"Warning: Unexpected header in {DEV_RERATED_FILE}. Re-writing.")
                         write_rerated_header() # Attempt to fix header
                    else:
                        for row in reader:
                            if row and len(row) >= 1:
                                rerated.add(row[0])
                except StopIteration:
                     if not os.path.exists(DEV_RERATED_FILE) or os.path.getsize(DEV_RERATED_FILE) == 0:
                          write_rerated_header()
        except (IOError, csv.Error, IndexError) as e:
            print(f"Warning: Could not read or parse re-rated file {DEV_RERATED_FILE}. Treating as empty. Error: {e}")
            write_rerated_header()
    else:
        # Ensure header exists if file doesn't exist yet
        write_rerated_header()
    return rerated

def write_rerated_header():
    """Writes the header row to the dev_rerated.csv file if needed."""
    write_header = True
    if os.path.exists(DEV_RERATED_FILE) and os.path.getsize(DEV_RERATED_FILE) > 10:
         try:
             with open(DEV_RERATED_FILE, 'r', newline='') as f:
                 reader = csv.reader(f)
                 header = next(reader)
                 if header == ['filename', 'new_label']:
                     write_header = False
         except (IOError, StopIteration, csv.Error):
             pass

    if write_header:
        try:
            with open(DEV_RERATED_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'new_label'])
        except IOError as e:
            print(f"Error: Could not write header to re-rated file {DEV_RERATED_FILE}. Error: {e}")

def get_next_image_to_rerate(all_dev_images, already_rerated_images):
    """Finds the next image from the dev set that hasn't been re-rated."""
    remaining_to_rerate = [img for img in all_dev_images if img not in already_rerated_images]

    if remaining_to_rerate:
        print(f"Selecting from {len(remaining_to_rerate)} dev images remaining to re-rate.")
        return random.choice(remaining_to_rerate)
    else:
        print("All dev images have been re-rated.")
        return None


# --- Flask Routes ---
# Access the mode set in the main guard
mode = app.config.get('LABELER_MODE', 'label')

@app.route('/', methods=['GET', 'POST'])
def index():
    # --- POST Request Logic ---
    if request.method == 'POST':
        filename = request.form.get('filename')
        action = request.form.get('label') # Input name is 'label' in the form

        if not filename or not action:
            flash("Error: Missing filename or action in submission.", "error")
            return redirect(url_for('index'))

        # --- Handle POST based on mode ---
        if mode == 'rerate':
            # In re-rate mode, only accept the 4 label values
            if action in LABEL_MAP:
                label_int = LABEL_MAP[action]
                try:
                    # Append to re-rated CSV
                    with open(DEV_RERATED_FILE, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([filename, label_int])
                    flash(f"Re-rated '{filename}' as '{action}' ({label_int}).", "success")
                except IOError as e:
                    flash(f"Error writing re-rated label for {filename}: {e}", "error")
                except Exception as e:
                     flash(f"An unexpected error occurred writing re-rated label: {e}", "error")
            else:
                # Ignore skip/flag/invalid actions in re-rate mode POST
                flash(f"Invalid action '{action}' in re-rate mode. Only labels (H,J,K,L) are accepted.", "warning")

        elif mode == 'label':
            # Original labeling mode POST logic
            if action not in VALID_ACTIONS:
                flash(f"Error: Invalid action '{action}' received.", "error")
                return redirect(url_for('index'))

            global last_action_info # Declare intent to modify global variable
            last_action_info = {'action': action, 'filename': filename, 'value': None}

            if action in LABEL_MAP:
                label_int = LABEL_MAP[action]
                last_action_info['value'] = label_int
                try:
                    with open(LABELS_FILE, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([filename, label_int])
                    flash(f"Labeled '{filename}' as '{action}' ({label_int}).", "success")
                except IOError as e:
                    flash(f"Error writing label for {filename}: {e}", "error")
                    last_action_info = None
                except Exception as e:
                     flash(f"An unexpected error occurred writing label: {e}", "error")
                     last_action_info = None

            elif action == 'skip':
                append_line_to_file(SKIPPED_FILE, filename)
                flash(f"Skipped '{filename}'.", "info")
                last_action_info['value'] = None # Ensure value is None for skip/flag

            elif action == 'flag':
                append_line_to_file(FLAGGED_FILE, filename)
                flash(f"Flagged '{filename}'.", "warning")
                last_action_info['value'] = None # Ensure value is None for skip/flag

        # Redirect to GET for the next image in either mode
        return redirect(url_for('index'))

    # --- GET Request Logic ---
    next_image = None
    progress_text = "Initializing..."
    progress_detail = ""
    is_done = False
    page_title = "Label the Face" # Default title

    if mode == 'rerate':
        page_title = "Re-rate Dev Set Image"
        all_dev_images = get_dev_images_to_rerate()
        if not all_dev_images:
            # Error message already flashed by get_dev_images_to_rerate
             return render_template('labeler.html', mode=mode, title=page_title, image_file=None, progress_text="Error loading dev set.", is_done=True)

        already_rerated = get_rerated_images()
        next_image = get_next_image_to_rerate(all_dev_images, already_rerated)

        total_dev_images = len(all_dev_images)
        rerated_count = len(already_rerated)
        is_done = next_image is None

        progress_text = f"Re-rating Dev Set ({rerated_count} / {total_dev_images} completed)"
        if is_done:
            progress_detail = "Re-rating complete!"
        else:
            progress_detail = f"{total_dev_images - rerated_count} images remaining."

    elif mode == 'label':
        page_title = "Label the Face"
        all_images = get_all_images()
        if not all_images:
             flash(f"No images found. Ensure '{TRAINING_LIST_FILE}' exists and is not empty. "
                   f"Run generate_training_list.py if needed.", "warning")
             return render_template('labeler.html', mode=mode, title=page_title, image_file=None, progress_text="No images found.", is_done=False)

        labeled_images, skipped_images, flagged_images = get_processed_images()
        next_image = get_next_image_to_label(all_images, labeled_images, skipped_images, flagged_images)

        total_images = len(all_images)
        labeled_count = len(labeled_images)
        processed_count = len(labeled_images | skipped_images | flagged_images)
        remaining_unseen = len(all_images) - processed_count
        remaining_skippable = len(skipped_images - (labeled_images | flagged_images))

        progress_text = f"Labeled: {labeled_count} | Skipped: {len(skipped_images)} | Flagged: {len(flagged_images)} | Total Images: {total_images}"
        progress_detail = f"Remaining: {remaining_unseen} unseen, {remaining_skippable} skippable."

        is_done = next_image is None

        if is_done and total_images > 0:
            progress_text = f"All {total_images} images processed. Labeled: {labeled_count}, Skipped: {len(skipped_images)}, Flagged: {len(flagged_images)}."
            progress_detail = "Labeling complete!"

    # --- Common GET Logic ---
    image_path = None
    if next_image:
        image_path = url_for('serve_celeba_image', filename=next_image)

    # Pass mode to template for conditional rendering
    return render_template('labeler.html',
                           mode=mode, # Pass the mode
                           title=page_title, # Pass the title
                           image_file=next_image,
                           image_path=image_path,
                           progress_text=progress_text,
                           progress_detail=progress_detail,
                           is_done=is_done)


@app.route('/undo', methods=['POST'])
def undo():
    """Reverts the last label, skip, or flag action (only in 'label' mode)."""
    # Access the mode
    mode = app.config.get('LABELER_MODE', 'label')
    if mode == 'rerate':
        flash("Undo is not available in re-rating mode.", "warning")
        return {"status": "error", "message": "Undo unavailable in re-rate mode"}, 403 # Forbidden

    global last_action_info # Declare intent to read and modify global variable

    if last_action_info is None:
        flash("Nothing to undo.", "warning")
        # Return an error status that the JS fetch can check
        return {"status": "error", "message": "Nothing to undo"}, 400

    action = last_action_info.get('action')
    filename = last_action_info.get('filename')

    if not action or not filename:
        flash("Error: Invalid undo state.", "error")
        last_action_info = None # Clear invalid state
        return {"status": "error", "message": "Invalid undo state"}, 500

    success = False
    if action in LABEL_MAP:
        print(f"Undoing label for {filename}")
        success = remove_last_line_matching(LABELS_FILE, filename, is_csv=True)
    elif action == 'skip':
        print(f"Undoing skip for {filename}")
        success = remove_last_line_matching(SKIPPED_FILE, filename, is_csv=False)
    elif action == 'flag':
        print(f"Undoing flag for {filename}")
        success = remove_last_line_matching(FLAGGED_FILE, filename, is_csv=False)
    else:
        flash(f"Error: Unknown action '{action}' in undo state.", "error")
        last_action_info = None # Clear invalid state
        return {"status": "error", "message": "Unknown action"}, 500

    if success:
        flash(f"Successfully undone '{action}' for {filename}.", "success")
        last_action_info = None # Clear undo state after successful undo
        return {"status": "success"}, 200
    else:
        # The removal function already flashed an error, but we return error status
        print(f"Failed to undo {action} for {filename} (file modification failed).")
        # Keep last_action_info so user might retry? Or clear it? Let's clear it.
        last_action_info = None
        return {"status": "error", "message": "File update failed during undo"}, 500


@app.route('/images/<filename>')
def serve_celeba_image(filename):
    """Serves images directly from the CelebA image directory."""
    # Basic security check: Ensure filename doesn't try to escape the directory
    if '..' in filename or filename.startswith('/'):
         from flask import abort
         abort(404) # Or 400 Bad Request

    # Check if the requested file exists in the training list for added safety
    # (Optional but recommended if training_imgs.txt is the definitive source)
    # all_training_images = get_all_images() # Consider caching this if performance is an issue
    # if filename not in all_training_images:
    #     from flask import abort
    #     abort(404)

    print(f"Serving image: {filename} from {CELEBA_IMAGE_DIR}") # Add logging
    try:
        return send_from_directory(CELEBA_IMAGE_DIR, filename)
    except FileNotFoundError:
         from flask import abort
         print(f"Error: Image file not found at {os.path.join(CELEBA_IMAGE_DIR, filename)}")
         abort(404)
    except Exception as e:
         from flask import abort
         print(f"Error serving file {filename}: {e}")
         abort(500) # Internal Server Error for other issues


# --- Static File Serving Setup ---
# For serving images, Flask needs a 'static' folder next to the app script,
# or configure a different static folder.
# Static URL path remains the same
app.static_url_path = '/static'


if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="CelebA Image Labeler/Re-rater")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['label', 'rerate'],
        default='label',
        help="Mode of operation: 'label' for initial labeling, 'rerate' for re-rating the dev set."
    )
    args = parser.parse_args()

    # Store mode in Flask app config for access in routes
    app.config['LABELER_MODE'] = args.mode
    print(f"--- Running in {args.mode.upper()} mode ---")

    # Ensure necessary files/headers are checked/created based on mode
    if args.mode == 'label':
        print("Checking/Creating label, skip, flag files...")
        get_processed_images() # Checks labels.csv, skipped.txt, flagged.txt
    elif args.mode == 'rerate':
        print("Checking/Creating dev_rerated.csv file...")
        get_rerated_images() # Checks dev_rerated.csv

    # Note: For development, debug=True is helpful. For production, set debug=False.
    # host='0.0.0.0' makes it accessible on your network
    print(f"Serving images from: {CELEBA_IMAGE_DIR}")
    print(f"Access the tool at: http://127.0.0.1:5000 or http://<your-ip>:5000")
    app.run(debug=True, host='0.0.0.0')
