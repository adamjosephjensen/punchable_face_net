# Project Tasks: CelebA Annoying Face Labeling - Initial Model




## Phase 2: Labeler Enhancements & Workflow Refinement (Based on prompt.md)

**UI & Interaction Changes:**

1.  **Implement Undo Functionality:**
    *   Add an "Undo" button to `templates/labeler.html`.
    *   Implement backend logic in `app/labeler.py` to store the *last* action (label, skip, flag) and its associated data (filename, previous state).
    *   Create a new route (e.g., `/undo`) or modify the main route's POST handler to revert the last action (e.g., remove the last row from `labels.csv` or update the status of the last image). *Constraint: Only one level of undo is required.*
    *   Add JavaScript in `templates/labeler.html` to trigger the undo action via the button click or the 'a' keypress.

2.  **Change Skip Shortcut:**
    *   Modify JavaScript in `templates/labeler.html` to change the keyboard shortcut for the "Skip" action to 's'.

3.  **Implement Labeling Time Limit (1.5s):**
    *   In `templates/labeler.html`, add JavaScript to start a 1.5-second timer when a new image is displayed.
    *   If the timer expires before the user labels, flags, or manually skips, automatically trigger a "skip" action for the current image.
    *   *Self-Correction:* The prompt implies auto-skipped images should be treated like manually skipped ones regarding persistence and re-display logic. No separate "auto_skip" status needed, just trigger the existing skip mechanism.
    *   Add comments in the JavaScript code explaining the rationale (capturing visceral reaction).

4.  **Implement Flagging Functionality:**
    *   Add a "Flag" button to `templates/labeler.html`.
    *   Modify the POST handler in `app/labeler.py` to accept a 'flag' action.
    *   Implement a mechanism to track flagged images (e.g., maintain a separate file `data/flagged.txt` or add a 'status' column to `labels.csv` - *Decision: Keep `labels.csv` clean, use a separate `flagged.txt` file for simplicity*). Write the filename to `data/flagged.txt` when flagged.
    *   Add JavaScript in `templates/labeler.html` to trigger the flag action via button click or the 'f' keypress.

5.  **Update Classification Scheme (4 Classes):**
    *   Modify `templates/labeler.html`:
        *   Replace "Annoying" / "Not Annoying" buttons with four new buttons: "VERY punchable", "Punchable", "NOT punchable", "VERY NOT punchable".
        *   Assign keyboard shortcuts: 'H', 'J', 'K', 'L' respectively.
    *   Modify `app/labeler.py`:
        *   Update the POST handler to accept the four new label values.
        *   Define a mapping for these labels (e.g., "VERY punchable": 3, "Punchable": 2, "NOT punchable": 1, "VERY NOT punchable": 0).
        *   Update the CSV writing logic to store these numerical values.
    *   Modify `get_labeled_images` and `write_labels_header` in `app/labeler.py` to reflect the new label column meaning in `labels.csv`. The header should remain 'filename', 'label', but the comment/understanding changes.
    *   *Note:* This change requires subsequent updates to data splitting and model training.

**Data Handling & Workflow Logic:**

6.  **Manage Skipped/Flagged Image Display:**
    *   Modify `get_labeled_images` in `app/labeler.py` to also read `data/skipped.txt` (needs creation similar to flagged.txt) and `data/flagged.txt`. Maintain sets of skipped and flagged filenames.
    *   Modify `get_next_image_to_label` in `app/labeler.py`:
        *   Prioritize selecting images that are *not* in the labeled set, *not* in the skipped set, and *not* in the flagged set.
        *   If no such images remain, *then* select an image from the skipped set that is not in the labeled or flagged sets.
        *   Flagged images should ideally never be shown again unless explicitly requested via a different mechanism (outside current scope).
    *   Modify the POST handler in `app/labeler.py` to write to `data/skipped.txt` when a skip action occurs (manual or timed). Ensure the undo action removes the filename from the correct file (`labels.csv`, `skipped.txt`, or `flagged.txt`).

7.  **Exclude Skipped/Flagged from Training Data:**
    *   Modify `split_data.py`: Before splitting, filter the DataFrame read from `labels.csv` to ensure it only contains labels (it should by design if skipped/flagged aren't written there). *Crucially*, ensure `split_data.py` does *not* read from `skipped.txt` or `flagged.txt`. The final `train.csv`, `dev.csv`, `test.csv` must only contain successfully labeled, non-flagged images.
    *   *(Self-Correction):* `dataset.py` reads from the split CSVs. If `split_data.py` correctly excludes skipped/flagged images, `dataset.py` doesn't need changes for this specific exclusion task.

**Model & Evaluation Updates:**

8.  **Adapt Model for 4 Classes:**
    *   Modify `train.py`: Change the final `nn.Linear` layer in the ResNet model definition from `nn.Linear(num_ftrs, 2)` to `nn.Linear(num_ftrs, 4)`.
    *   Update any print statements or logging related to the number of classes. `CrossEntropyLoss` and accuracy calculations should adapt automatically to 4 classes.

9.  **Update README:**
    *   Modify `README.md` to reflect the new 4-class labeling scheme, the updated keyboard shortcuts (H, J, K, L for labels; 's' for skip; 'f' for flag; 'a' for undo), the timer functionality, and the flagging feature.

**Proposals (Describe Systems):**

10. **Propose System for Intra-Rater Reliability:**
    *   Describe a system in `tasks.md` or a separate design document:
        *   Modify `get_next_image_to_label` to occasionally (e.g., 10% probability) present an image that has *already been labeled* by the current user (drawn from `labels.csv`).
        *   Store this second label, perhaps in a separate file (`data/reratings.csv`) with `filename`, `original_label`, `new_label`, `timestamp`.
        *   Outline a separate script (`calculate_reliability.py`) that reads `reratings.csv` and calculates Cohen's Kappa (κ).
        *   Suggest how to trigger this calculation (e.g., manually, or periodically) and how to report low agreement (κ < 0.7) (e.g., print a warning).

11. **Propose System for Estimating Optimal Performance:**
    *   Describe a system in `tasks.md` or a separate design document:
        *   After initial labeling and data splitting, create a dedicated mode or script.
        *   This mode presents *only* the images listed in `data/dev.csv` to the original rater, one by one, *without* showing the previously assigned label.
        *   Record these new labels for the dev set images (e.g., in `data/dev_rerated.csv`).
        *   Outline a script (`calculate_benchmark_accuracy.py`) to compare `data/dev.csv` and `data/dev_rerated.csv` and calculate accuracy and/or Kappa. This score represents the rater's self-consistency on unseen data, serving as a practical upper bound for model performance.
