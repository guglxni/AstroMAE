import os
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PreprocessedAstroDataset(Dataset):
    """Loads preprocessed tensors (.pt files) from a directory structure.

    Assumes root_dir contains subdirectories for each class,
    and each subdirectory contains .pt files corresponding to processed images.
    Example: root_dir/class_A/image1.pt, root_dir/class_B/image2.pt
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # Recursively find all .pt files within the root_dir
        self.tensor_paths = glob.glob(os.path.join(root_dir, '**', '*.pt'), recursive=True)

        if not self.tensor_paths:
            logging.warning(f"No .pt files found recursively in {root_dir}. Ensure preprocessing was run and data exists.")
        else:
            logging.info(f"Found {len(self.tensor_paths)} preprocessed tensor files in {root_dir}.")

        # No transforms needed here as tensors are already processed
        # self.transform = None # Explicitly setting to None

    def __len__(self):
        return len(self.tensor_paths)

    def __getitem__(self, idx):
        tensor_path = self.tensor_paths[idx]
        try:
            # Load the pre-saved tensor
            tensor = torch.load(tensor_path)
            # Ensure tensor is float32, as expected by most models
            return tensor.float()
        except Exception as e:
            logging.error(f"Error loading tensor {tensor_path}: {e}")
            # Return a dummy tensor of expected shape if loading fails
            # Assuming shape (C, H, W) = (3, 128, 128) based on preprocessing
            # Adjust if the preprocessing dimensions change
            return torch.zeros((3, 128, 128))

def create_dataloader(dataset_dir, batch_size, num_workers=0, image_size=128):
    """Creates a DataLoader for the preprocessed dataset.

    Args:
        dataset_dir (str): Path to the directory containing processed tensors (e.g., './data/processed').
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.
        image_size (int): Expected image size (used for potential error handling, not transformation).
                         Kept for consistency but not actively used for loading .pt files.
    """
    dataset = PreprocessedAstroDataset(root_dir=dataset_dir)
    if len(dataset) == 0:
        logging.error(f"Dataset at {dataset_dir} is empty. Please check the directory and ensure preprocessing ran correctly.")
        return None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True, # Set to True if using GPU
        drop_last=True # Good practice for training, esp. with batch norm
    )
    logging.info(f"DataLoader created for {dataset_dir} with batch size {batch_size}.")
    return dataloader

# Example usage (optional, for testing)
if __name__ == '__main__':
    # Assumes you are running this from the project root (AstroMAE/)
    processed_dir = './data/processed'
    batch_size = 32

    # Check if the processed directory exists
    if not os.path.isdir(processed_dir):
        print(f"Error: Processed data directory '{processed_dir}' not found.")
        print("Please run the download and preprocess scripts first:")
        print("  python data/download.py")
        print("  python data/preprocess.py")
    else:
        dataloader = create_dataloader(processed_dir, batch_size=batch_size)

        if dataloader:
            print(f"Successfully created DataLoader. Number of batches: {len(dataloader)}")
            # Iterate over one batch to check
            try:
                for i, batch in enumerate(dataloader):
                    print(f"Batch {i+1} shape: {batch.shape}, dtype: {batch.dtype}")
                    # Check tensor value range (should be [0, 1] from ToTensor)
                    print(f"Batch {i+1} min value: {batch.min()}, max value: {batch.max()}")
                    break # Only check the first batch
            except Exception as e:
                print(f"Error iterating through DataLoader: {e}")
        else:
            print("Failed to create DataLoader.")