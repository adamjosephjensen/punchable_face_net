import os
import csv
import random
from flask import Flask, render_template, request, redirect, url_for, flash

# --- Configuration ---
IMAGE_DIR = os.path.abspath('./data/images') # Adjust if your structure differs
LABELS_FILE = os.path.abspath('./data/labels.csv')
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
BATCH_SIZE = 10 # How many images to label before showing progress

# --- Flask App Setup ---
# Point Flask to the templates directory at the project root
app = Flask(__name__, template_folder=os.path.abspath('../templates'))
app.secret_key = 'super secret key' # Change this for production use

# --- Helper Functions ---
def get_all_images():
    """Gets a list of all image filenames in the IMAGE_DIR."""
    images = []
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Image directory not found at {IMAGE_DIR}")
        return images
    for fname in os.listdir(IMAGE_DIR):
        if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
            images.append(fname)
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
    progress_text = f"{labeled_count} / {total_images} images labeled."

    # Calculate percentage for the progress bar
    progress_percent = 0
    if total_images > 0:
        progress_percent = round((labeled_count / total_images) * 100)

    image_path = None
    if next_image:
        # Construct URL path relative to static folder if using Flask's static serving
        # For simplicity here, assuming direct access or a route to serve images
        # A better approach might be a dedicated /image/<filename> route
        image_path = url_for('static', filename=f'images/{next_image}') # Assumes images are in static/images

    # Use just the filename, Flask searches in the configured template_folder
    return render_template('labeler.html',
                           image_file=next_image,
                           image_path=image_path,
                           progress_text=progress_text,
                           progress_percent=progress_percent) # Add progress_percent here


# --- Template ---
# You would typically put this in a templates/labeler.html file
LABELER_HTML = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Image Labeler</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        .container { max-width: 600px; margin: auto; text-align: center; }
        img { max-width: 100%; height: auto; margin-bottom: 20px; display: block; margin-left: auto; margin-right: auto; }
        .controls button { padding: 10px 20px; margin: 5px; font-size: 16px; cursor: pointer; }
        .progress { margin-top: 20px; font-size: 14px; color: grey; }
        .flash { padding: 10px; margin-bottom: 15px; border-radius: 4px; }
        .flash.error { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .flash.warning { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
        .flash.success { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        /* Key hints */
        .key-hints { margin-top: 15px; font-size: 0.9em; color: #555; }
        .key-hints kbd {
            display: inline-block;
            padding: 3px 5px;
            font: 11px SFMono-Regular,Consolas,Liberation Mono,Menlo,monospace;
            line-height: 10px;
            color: #444d56;
            vertical-align: middle;
            background-color: #fcfcfc;
            border: solid 1px #c6cbd1;
            border-bottom-color: #959da5;
            border-radius: 3px;
            box-shadow: inset 0 -1px 0 #959da5;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Label the Face</h1>

        <!-- Flash messages -->
        {% with messages = get_flashed_messages(with_categories=true) %}
          {% if messages %}
            {% for category, message in messages %}
              <div class="flash {{ category }}">{{ message }}</div>
            {% endfor %}
          {% endif %}
        {% endwith %}

        <p class="progress">{{ progress_text }}</p>

        {% if image_file %}
            <img src="{{ image_path }}" alt="Image to label" id="labeling-image">
            <form method="post" action="{{ url_for('index') }}" id="label-form">
                <input type="hidden" name="filename" value="{{ image_file }}">
                <input type="hidden" name="label" id="label-input">
                <div class="controls">
                    <button type="button" onclick="submitLabel('annoying')">Annoying (H)</button>
                    <button type="button" onclick="submitLabel('not_annoying')">Not Annoying (L)</button>
                    <button type="button" onclick="submitLabel('skip')">Skip (S)</button>
                </div>
                 <div class="key-hints">
                    Use keyboard shortcuts: <kbd>H</kbd> for Annoying, <kbd>L</kbd> for Not Annoying, <kbd>S</kbd> for Skip
                </div>
            </form>
        {% else %}
            <p>No more images to label, or no images found!</p>
        {% endif %}
    </div>

    <script>
        function submitLabel(labelValue) {
            document.getElementById('label-input').value = labelValue;
            document.getElementById('label-form').submit();
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', function(event) {
            // Ensure we don't trigger shortcuts if user is typing in an input field (none here, but good practice)
            if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
                return;
            }

            // Check if the image and form exist
            if (!document.getElementById('labeling-image')) {
                return; // No image currently displayed
            }

            switch(event.key.toUpperCase()) {
                case 'H':
                    submitLabel('annoying');
                    break;
                case 'L':
                    submitLabel('not_annoying');
                    break;
                case 'S':
                    submitLabel('skip');
                    break;
            }
        });
    </script>
</body>
</html>
"""

# Create templates directory if it doesn't exist
if not os.path.exists('templates'):
    os.makedirs('templates')

# Write the HTML content to templates/labeler.html
with open('templates/labeler.html', 'w') as f:
    f.write(LABELER_HTML)

# --- Static File Serving Setup ---
# For serving images, Flask needs a 'static' folder next to the app script,
# or configure a different static folder.
# Let's assume images are in ../data/images relative to this script.
# We need to tell Flask where to find them.
app.static_folder = os.path.abspath('../data')
app.static_url_path = '/static' # URL path to access static files


if __name__ == '__main__':
    # Make sure labels file exists with header before starting
    get_labeled_images()
    # Note: For development, debug=True is helpful.
    # For production or sharing, set debug=False.
    # host='0.0.0.0' makes it accessible on your network
    print(f"Serving images from: {app.static_folder}")
    print(f"Access the labeler at: http://127.0.0.1:5000 or http://<your-ip>:5000")
    app.run(debug=True, host='0.0.0.0')
