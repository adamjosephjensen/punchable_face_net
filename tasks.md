# Project Tasks: CelebA Annoying Face Labeling - Initial Model

This file outlines the steps to get the first baseline model trained for the subjective "annoying
face" classification task using transfer learning.

## Phase 1: Data Preparation and Baseline Model Training

1.  **[DONE] Label Initial Data:** Labeled 1000 images using the Flask labeler. (`data/labels.csv`)
2.  **[TODO] Create Data Split Script (`split_data.py`):**
    *   Read `data/labels.csv`.
    *   Use `sklearn.model_selection.train_test_split` (stratified) to split data into Train (70%),
Validation (20%), and Test (10%) sets.
    *   Save the splits into `data/train.csv`, `data/dev.csv`, `data/test.csv` (containing
'filename' and 'label' columns).
3.  **[TODO] Run Data Split Script:** Execute `python split_data.py` to generate the CSV files.
4.  **[TODO] Create PyTorch Dataset Class (`dataset.py` or similar):**
    *   Define a class inheriting from `torch.utils.data.Dataset`.
    *   `__init__`: Takes CSV path, CelebA image directory path, and transforms. Reads CSV.
    *   `__len__`: Returns dataset size.
    *   `__getitem__`: Loads image by filename, applies transforms, returns (image_tensor,
label_tensor).
5.  **[TODO] Create Main Training Script (`train.py`):**
    *   **Imports:** `torch`, `torchvision`, `pandas`, `argparse`,
`torch.utils.tensorboard.SummaryWriter`, your `Dataset` class.
    *   **Argument Parsing:** Define arguments for `--image_dir`, `--csv_train`, `--csv_val`,
`--epochs`, `--batch_size`, `--lr` (learning rate), `--log_dir` (for TensorBoard),
`--checkpoint_dir`.
    *   **Device Setup:** Set device to CUDA if available, else CPU.
    *   **Transforms:** Define training transforms (Resize, RandomHorizontalFlip, ToTensor,
Normalize) and validation transforms (Resize, ToTensor, Normalize).
    *   **Datasets & DataLoaders:** Instantiate `CelebADataset` for train and validation sets.
Create `DataLoader` for each (shuffle train loader).
    *   **Model Definition:**
        *   Load pre-trained `torchvision.models.resnet18(pretrained=True)`.
        *   Freeze all parameters (`param.requires_grad = False`).
        *   Replace `model.fc` with a new `torch.nn.Linear` layer mapping `in_features` to 2 output
classes.
        *   Move model to the configured `device`.
    *   **Loss & Optimizer:**
        *   `criterion = torch.nn.CrossEntropyLoss()`
        *   `optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)` (Note: only optimizing
the new layer).
    *   **TensorBoard Setup:** `writer = SummaryWriter(args.log_dir)`
    *   **Training Loop (per epoch):**
        *   `model.train()`
        *   Loop through training batches.
        *   Move data to `device`.
        *   Zero gradients, forward pass, calculate loss, backward pass, optimizer step.
        *   Accumulate training loss and accuracy for the epoch.
    *   **Validation Loop (per epoch):**
        *   `model.eval()`
        *   `with torch.no_grad():`
        *   Loop through validation batches.
        *   Move data to `device`.
        *   Forward pass, calculate loss.
        *   Calculate validation accuracy for the epoch.
        *   Accumulate validation loss and accuracy for the epoch.
    *   **Logging:** After each epoch, log `train_loss`, `train_acc`, `val_loss`, `val_acc` to
TensorBoard using `writer.add_scalar(...)`. Print epoch results to console.
    *   **Checkpointing (Optional but Recommended):** Save the model weights
(`torch.save(model.state_dict(), ...)`) if the current epoch's validation accuracy is the best seen
so far.
    *   **Cleanup:** `writer.close()` after the training loop finishes.
6.  **[TODO] Run Training Script:** Execute `python train.py --image_dir
/path/to/celeba/img_align_celeba --csv_train data/train.csv --csv_val data/dev.csv ...` (fill in
other args).
7.  **[TODO] Monitor with TensorBoard:** While training runs, execute `tensorboard --logdir runs`
(or your specified log directory) in a separate terminal and open the provided URL in your browser.
8.  **[TODO] Analyze Results:** Examine the learning curves in TensorBoard. Evaluate the final
validation accuracy. Decide on next steps based on performance (e.g., tune hyperparameters, unfreeze
more layers, collect more data).
9.  **[TODO] Evaluate on Test Set:** *Only once*, after all tuning and model selection is done using
the validation set, evaluate the final chosen model on the `test.csv` data to get an unbiased
performance estimate.
