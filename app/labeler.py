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
# IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'} # No longer needed when reading from list
BATCH_SIZE = 10 # How many images to label before showing progress

# --- Flask App Setup ---
# Calculate paths relative to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.abspath(os.path.join(script_dir, '..', 'templates'))
static_dir = os.path.abspath(os.path.join(script_dir, '..', 'data'))

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.secret_key = 'super secret key' # Change this for production use

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

def get_labeled_images():
    """Gets a set of filenames that have already been labeled."""
    labeled = set()
    if os.path.exists(LABELS_FILE):
        try:
            with open(LABELS_FILE, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader) # Skip header
                for row in reader:
                    if row: # Avoid issues with blank rows
                        labeled.add(row[0])
        except (FileNotFoundError, StopIteration, IndexError, csv.Error) as e:
            print(f"Warning: Could not read or parse labels file {LABELS_FILE}. Starting fresh. Error: {e}")
            # If file is corrupt or empty, treat as no labels exist
            if os.path.exists(LABELS_FILE):
                 # Optionally backup corrupt file here before overwriting
                 pass
            # Ensure header exists for writing later
            write_labels_header()
    else:
        # Create the file with header if it doesn't exist
        write_labels_header()

    return labeled

def write_labels_header():
     """Writes the header row to the CSV file."""
     try:
        with open(LABELS_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'label']) # label: 1=annoying, 0=not annoying
     except IOError as e:
         print(f"Error: Could not write header to labels file {LABELS_FILE}. Error: {e}")


def get_next_image_to_label(all_images, labeled_images):
    """Finds the next unlabeled image."""
    unlabeled = [img for img in all_images if img not in labeled_images]
    if not unlabeled:
        return None
    return random.choice(unlabeled) # Pick a random unlabeled image

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        filename = request.form.get('filename')
        label = request.form.get('label') # 'annoying' or 'not_annoying' or 'skip'

        if not filename or not label:
             flash("Error: Missing filename or label in submission.", "error")
             return redirect(url_for('index'))

        if label != 'skip':
            # Convert label to 1 or 0
            label_int = 1 if label == 'annoying' else 0
            try:
                # Append to CSV
                with open(LABELS_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([filename, label_int])
            except IOError as e:
                flash(f"Error writing label for {filename}: {e}", "error")
                # Consider how to handle write errors - maybe retry?
            except Exception as e:
                 flash(f"An unexpected error occurred writing label: {e}", "error")


        # Redirect to GET to show the next image
        return redirect(url_for('index'))

    # --- GET Request Logic ---
    all_images = get_all_images()
    if not all_images:
         flash(f"No images found in {IMAGE_DIR}. Please add images.", "warning")
         # Use just the filename, Flask searches in the configured template_folder
         return render_template('labeler.html', image_file=None, progress_text="No images found.")

    labeled_images = get_labeled_images()
    next_image = get_next_image_to_label(all_images, labeled_images)

    total_images = len(all_images)
    labeled_count = len(labeled_images)
    # Update progress text even when done
    progress_text = f"{labeled_count} / {total_images} images labeled."
    if next_image is None and total_images > 0: # Check if done and there were images initially
        progress_text += " - All images labeled!"

    # Calculate percentage for the progress bar
    progress_percent = 0
    if total_images > 0:
        progress_percent = round((labeled_count / total_images) * 100)

    image_path = None
    image_path = None
    if next_image:
        # Use the new route to serve images directly from CelebA dir
        image_path = url_for('serve_celeba_image', filename=next_image)

    # Add a flag to indicate completion status
    is_done = next_image is None and total_images > 0 # True if no next image AND there were images

    # Use just the filename, Flask searches in the configured template_folder
    return render_template('labeler.html',
                           image_file=next_image,
                           image_path=image_path,
                           progress_text=progress_text,
                           progress_percent=progress_percent,
                           is_done=is_done) # Pass the completion flag


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
    # Make sure labels file exists with header before starting
    get_labeled_images()
    # Note: For development, debug=True is helpful.
    # For production or sharing, set debug=False.
    # host='0.0.0.0' makes it accessible on your network
    print(f"Serving images from: {app.static_folder}")
    print(f"Access the labeler at: http://127.0.0.1:5000 or http://<your-ip>:5000")
    app.run(debug=True, host='0.0.0.0')
