import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import mean_squared_error, cohen_kappa_score # Added for regression metrics
import numpy as np # Added for Kappa calculation
from dataset import CelebADataset # Assuming dataset.py is in the same directory

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Train a simple classifier on CelebA subset.')
parser.add_argument('--image_dir', type=str,
                    default='/Users/adamjensen/Documents/celebA/CelebA/Img/img_align_celeba', # Default path
                    help='Path to the root directory containing CelebA images (default: specified path)')
parser.add_argument('--csv_train', type=str, default='data/train.csv',
                    help='Path to the training CSV file')
parser.add_argument('--csv_val', type=str, default='data/dev.csv',
                    help='Path to the validation CSV file')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for training and validation')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate for the optimizer')
parser.add_argument('--log_dir', type=str, default='runs/celeba_annoying_resnet18_run1',
                    help='Directory for TensorBoard logs')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                    help='Directory to save model checkpoints')
parser.add_argument('--num_workers', type=int, default=2, # Sensible default, adjust based on system
                    help='Number of worker processes for DataLoader')
args = parser.parse_args()

# --- Main Training Function ---
def train_model():
    print("Starting training process...")
    print(f"Configuration: {vars(args)}")

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Transforms ---
    # Standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    image_size = 224 # Standard for ResNet

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    # --- Datasets & DataLoaders ---
    try:
        print(f"Loading training data from: {args.csv_train}")
        train_dataset = CelebADataset(csv_file=args.csv_train, image_dir=args.image_dir, transform=train_transform)
        print(f"Loading validation data from: {args.csv_val}")
        val_dataset = CelebADataset(csv_file=args.csv_val, image_dir=args.image_dir, transform=val_transform)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"Error loading datasets: {e}")
        print("Please ensure CSV files and image directory are correct.")
        return # Exit if data cannot be loaded

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True if device == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True if device == 'cuda' else False)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

    # --- Model Definition (Transfer Learning with ResNet18) ---
    print("Loading pre-trained ResNet18 model...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) # Use updated weights API

    # Freeze all base layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer for regression (output 1 value)
    num_ftrs = model.fc.in_features
    num_outputs = 1 # Regression output
    model.fc = nn.Linear(num_ftrs, num_outputs)
    print(f"Replaced final layer for regression. Output units: {num_outputs}")

    model = model.to(device)

    # --- Loss & Optimizer ---
    criterion = nn.MSELoss() # Use Mean Squared Error for regression
    # Only optimize the parameters of the new final layer
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    print(f"Optimizer: Adam, LR: {args.lr}. Loss: MSE. Optimizing only the final layer.")

    # --- TensorBoard Setup ---
    print(f"Setting up TensorBoard logging to: {args.log_dir}")
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    # --- Checkpoint Setup ---
    print(f"Checkpoints will be saved to: {args.checkpoint_dir}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    # Save based on lowest validation MSE (the direct optimization target)
    best_val_mse = float('inf')
    best_model_path = os.path.join(args.checkpoint_dir, 'best_model_mse.pth') # Rename checkpoint file

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0 # This will be MSE loss
        total_train_samples = 0

        for i, (inputs, labels) in enumerate(train_loader):
            # Ensure labels are float and have the right shape for MSELoss (batch_size, 1)
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device) # Convert to float and add dimension

            optimizer.zero_grad()

            outputs = model(inputs) # Output shape will be (batch_size, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            total_train_samples += inputs.size(0)

            # Optional: Print progress within epoch
            # if (i + 1) % 10 == 0: # Print every 10 batches
            #     print(f'  Batch {i+1}/{len(train_loader)} Train MSE: {loss.item():.4f}')

        epoch_train_loss = running_train_loss / total_train_samples # Now represents average MSE

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0 # MSE loss
        running_val_corrects = 0 # Accuracy based on rounded predictions
        total_val_samples = 0
        all_val_preds_rounded = []
        all_val_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                # Ensure labels are float and have the right shape for MSELoss
                labels_float = labels.float().unsqueeze(1).to(device)
                # Keep original int labels for accuracy/kappa calculation
                labels_int = labels.long().to(device)

                outputs = model(inputs) # Shape (batch_size, 1)
                loss = criterion(outputs, labels_float)

                # --- Calculate metrics ---
                running_val_loss += loss.item() * inputs.size(0)

                # Round predictions to nearest integer (0, 1, 2, 3) for accuracy/kappa
                # Clamp predictions to be within the valid label range before rounding
                preds_clamped = torch.clamp(outputs.squeeze(), min=0, max=3) # Max label is 3
                preds_rounded = torch.round(preds_clamped).long()

                running_val_corrects += torch.sum(preds_rounded == labels_int.data)
                total_val_samples += inputs.size(0)

                # Store all predictions and labels for epoch-level Kappa calculation
                all_val_preds_rounded.extend(preds_rounded.cpu().numpy())
                all_val_labels.extend(labels_int.cpu().numpy())


        epoch_val_loss = running_val_loss / total_val_samples # Average validation MSE
        epoch_val_acc_rounded = running_val_corrects.double() / total_val_samples

        # Calculate Quadratic Weighted Kappa
        # Ensure lists are not empty before calculating kappa
        epoch_val_kappa = 0.0
        if all_val_labels and all_val_preds_rounded:
             try:
                 # Use weights='quadratic' for ordinal agreement
                 epoch_val_kappa = cohen_kappa_score(all_val_labels, all_val_preds_rounded, weights='quadratic')
             except Exception as e:
                 print(f"Warning: Could not calculate Kappa. Error: {e}")
                 epoch_val_kappa = 0.0 # Assign a default value or handle as needed
        else:
             print("Warning: No validation labels/predictions found to calculate Kappa.")


        # --- Logging ---
        writer.add_scalar('Loss_MSE/train', epoch_train_loss, epoch)
        writer.add_scalar('Loss_MSE/validation', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy_Rounded/validation', epoch_val_acc_rounded, epoch)
        writer.add_scalar('Kappa_QuadraticWeighted/validation', epoch_val_kappa, epoch)


        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} Summary | Time: {epoch_time:.2f}s")
        print(f"  Train MSE: {epoch_train_loss:.4f}")
        print(f"  Val MSE:   {epoch_val_loss:.4f} | Val Acc (Rounded): {epoch_val_acc_rounded:.4f} | Val Weighted Kappa: {epoch_val_kappa:.4f}")

        # --- Checkpointing (Save based on lowest validation MSE) ---
        if epoch_val_loss < best_val_mse:
            best_val_mse = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best validation MSE: {best_val_mse:.4f}. Saved model to {best_model_path}")

    # --- End of Training ---
    total_training_time = time.time() - start_time
    print("\n--- Training Finished ---")
    print(f"Total Training Time: {total_training_time:.2f}s")
    print(f"Lowest Validation MSE: {best_val_mse:.4f}") # Updated metric
    print(f"Best model (lowest MSE) saved to: {best_model_path}")
    print(f"TensorBoard logs saved to: {args.log_dir}")

    writer.close()
    print("TensorBoard writer closed.")

# --- Main Guard ---
if __name__ == "__main__":
    train_model()
