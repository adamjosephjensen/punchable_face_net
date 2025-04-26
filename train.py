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
# Removed regression metrics: mean_squared_error, cohen_kappa_score
# Removed numpy as it was only needed for Kappa
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

    # Replace the final fully connected layer for 4-class classification
    num_ftrs = model.fc.in_features
    num_classes = 4 # Classification output
    model.fc = nn.Linear(num_ftrs, num_classes)
    print(f"Replaced final layer for classification. Output classes: {num_classes}")

    model = model.to(device)

    # --- Loss & Optimizer ---
    criterion = nn.CrossEntropyLoss() # Use Cross Entropy for classification
    # Only optimize the parameters of the new final layer
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    print(f"Optimizer: Adam, LR: {args.lr}. Loss: CrossEntropyLoss. Optimizing only the final layer.")

    # --- TensorBoard Setup ---
    print(f"Setting up TensorBoard logging to: {args.log_dir}")
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    # --- Checkpoint Setup ---
    print(f"Checkpoints will be saved to: {args.checkpoint_dir}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    # Save based on highest validation accuracy
    best_val_acc = 0.0
    best_model_path = os.path.join(args.checkpoint_dir, 'best_model_acc.pth') # Checkpoint file based on accuracy

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0 # CrossEntropyLoss
        running_train_corrects = 0
        total_train_samples = 0

        for i, (inputs, labels) in enumerate(train_loader):
            # Labels should be LongTensor for CrossEntropyLoss
            inputs = inputs.to(device)
            labels = labels.to(device) # Should be LongTensor from dataset

            optimizer.zero_grad()

            outputs = model(inputs) # Output shape will be (batch_size, num_classes)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1) # Get predicted class index

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            running_train_corrects += torch.sum(preds == labels.data)
            total_train_samples += inputs.size(0)

            # Optional: Print progress within epoch
            # if (i + 1) % 10 == 0: # Print every 10 batches
            #     print(f'  Batch {i+1}/{len(train_loader)} Train Loss: {loss.item():.4f}')

        epoch_train_loss = running_train_loss / total_train_samples
        epoch_train_acc = running_train_corrects.double() / total_train_samples

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0 # CrossEntropyLoss
        running_val_corrects = 0
        total_val_samples = 0
        # Removed Kappa-related lists: all_val_preds_rounded, all_val_labels

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device) # Should be LongTensor

                outputs = model(inputs) # Shape (batch_size, num_classes)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1) # Get predicted class index

                # --- Calculate metrics ---
                running_val_loss += loss.item() * inputs.size(0)
                running_val_corrects += torch.sum(preds == labels.data)
                total_val_samples += inputs.size(0)

                # Removed Kappa calculation logic

        epoch_val_loss = running_val_loss / total_val_samples
        epoch_val_acc = running_val_corrects.double() / total_val_samples

        # Removed Kappa calculation

        # --- Logging ---
        writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_train_acc, epoch)
        writer.add_scalar('Loss/validation', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/validation', epoch_val_acc, epoch)
        # Removed Kappa logging

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} Summary | Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        print(f"  Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.4f}") # Removed Kappa from print

        # --- Checkpointing (Save based on highest validation accuracy) ---
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best validation accuracy: {best_val_acc:.4f}. Saved model to {best_model_path}")

    # --- End of Training ---
    total_training_time = time.time() - start_time
    print("\n--- Training Finished ---")
    print(f"Total Training Time: {total_training_time:.2f}s")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}") # Back to accuracy
    print(f"Best model (highest accuracy) saved to: {best_model_path}")
    print(f"TensorBoard logs saved to: {args.log_dir}")

    writer.close()
    print("TensorBoard writer closed.")

# --- Main Guard ---
if __name__ == "__main__":
    train_model()
