import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class CelebADataset(Dataset):
    """
    Custom PyTorch Dataset for loading CelebA images based on a CSV file.
    Assumes the CSV file has 'filename' and 'label' columns.
    """
    def __init__(self, csv_file, image_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
                               (e.g., 'data/train.csv')
            image_dir (string): Directory with all the images.
                                (e.g., '/path/to/celeba/img_align_celeba')
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
        """
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        if not os.path.isdir(image_dir):
             raise NotADirectoryError(f"Image directory not found or not a directory: {image_dir}")

        self.labels_frame = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        print(f"Initialized dataset from {csv_file}. Found {len(self.labels_frame)} samples.")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.labels_frame)

    def __getitem__(self, idx):
        """
        Fetches the sample (image and label) at the given index.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            tuple: (image, label) where image is the transformed image tensor,
                   and label is the corresponding label tensor.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image filename and label from the dataframe
        img_name = self.labels_frame.iloc[idx, 0] # Assumes 'filename' is the first column
        label = self.labels_frame.iloc[idx, 1]    # Assumes 'label' is the second column

        # Construct full image path
        img_path = os.path.join(self.image_dir, img_name)

        try:
            # Load image using PIL
            image = Image.open(img_path).convert('RGB') # Ensure image is RGB
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path} (referenced in {self.labels_frame.attrs.get('csv_path', 'CSV')})")
            # Handle missing image: return None or raise error?
            # For now, let's raise an error to stop training if data is missing.
            raise FileNotFoundError(f"Image file not found: {img_path}")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Raise error for other loading issues too
            raise e

        # Apply transformations if provided
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Error applying transforms to image {img_path}: {e}")
                raise e

        # Convert label to tensor (long tensor for CrossEntropyLoss)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image, label_tensor

# Example usage (optional, for testing the dataset class directly)
if __name__ == '__main__':
    # --- Configuration for testing ---
    # !! IMPORTANT: Update these paths for your system !!
    TEST_CSV = 'data/dev.csv' # Use dev set for a quick test
    CELEBA_IMG_DIR_TEST = '/Users/adamjensen/Documents/celebA/CelebA/Img/img_align_celeba'

    # Define some basic transforms for testing
    from torchvision import transforms
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)), # Smaller resize for quick testing
        transforms.ToTensor()
    ])
    # --- End Configuration ---

    print("Testing CelebADataset...")
    if not os.path.exists(TEST_CSV):
        print(f"Test CSV file not found at {TEST_CSV}. Skipping dataset test.")
    elif not os.path.isdir(CELEBA_IMG_DIR_TEST):
         print(f"CelebA image directory not found at {CELEBA_IMG_DIR_TEST}. Skipping dataset test.")
    else:
        try:
            dataset = CelebADataset(csv_file=TEST_CSV,
                                    image_dir=CELEBA_IMG_DIR_TEST,
                                    transform=test_transform)

            print(f"Dataset length: {len(dataset)}")

            # Try getting the first sample
            if len(dataset) > 0:
                img, lbl = dataset[0]
                print("First sample loaded successfully:")
                print(f"  Image shape: {img.shape}") # Should be [3, 64, 64] with test_transform
                print(f"  Label: {lbl.item()}")     # Should be 0 or 1
            else:
                print("Dataset is empty, cannot fetch sample.")

        except FileNotFoundError as e:
            print(f"Dataset test failed: {e}")
        except NotADirectoryError as e:
             print(f"Dataset test failed: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during dataset test: {e}")

    print("Dataset test finished.")
