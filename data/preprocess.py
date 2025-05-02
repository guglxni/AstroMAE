# This script will handle image resizing and normalization for the downloaded galaxy images

import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import logging
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define directories
BASE_DIR = os.path.dirname(__file__)
RAW_DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, 'raw'))
PROCESSED_DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, 'processed'))

# Define image transformations
IMG_SIZE = 128
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(), # Converts PIL image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    # Normalization is often dataset-specific. For now, ToTensor scales to [0, 1].
    # If specific mean/std normalization is needed later, add transforms.Normalize(mean=[...], std=[...])
])

def preprocess_images(raw_dir, processed_dir, transform):
    """Loads images from raw_dir, applies transforms, and saves to processed_dir."""
    logging.info(f"Starting preprocessing from {raw_dir} to {processed_dir}")

    if not os.path.isdir(raw_dir):
        logging.error(f"Raw data directory not found: {raw_dir}")
        logging.error("Please run the download script first (e.g., python data/download.py)")
        return

    # Clean the processed directory before saving new files
    if os.path.exists(processed_dir):
        logging.warning(f"Processed directory {processed_dir} already exists. Removing it.")
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir)
    logging.info(f"Created empty processed directory: {processed_dir}")

    try:
        # Use ImageFolder to load images structured by class folders
        raw_dataset = ImageFolder(root=raw_dir)
        logging.info(f"Found {len(raw_dataset)} images in {len(raw_dataset.classes)} classes in {raw_dir}.")

        processed_count = 0
        # Process and save images, maintaining the class structure
        for class_name in raw_dataset.classes:
            raw_class_dir = os.path.join(raw_dir, class_name)
            processed_class_dir = os.path.join(processed_dir, class_name)
            os.makedirs(processed_class_dir, exist_ok=True)

            for img_name in os.listdir(raw_class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        img_path = os.path.join(raw_class_dir, img_name)
                        # Open image using PIL
                        img = Image.open(img_path).convert('RGB') # Ensure 3 channels

                        # Apply transformations
                        processed_tensor = transform(img)

                        # Save the processed tensor
                        # Keep the original filename but change extension to .pt
                        base_name = os.path.splitext(img_name)[0]
                        tensor_filename = f"{base_name}.pt"
                        tensor_path = os.path.join(processed_class_dir, tensor_filename)
                        torch.save(processed_tensor, tensor_path)
                        processed_count += 1
                    except Exception as e:
                        logging.error(f"Failed to process image {img_name} in {class_name}: {e}")

        logging.info(f"Successfully processed and saved {processed_count} images to {processed_dir}")

    except FileNotFoundError:
        logging.error(f"ImageFolder could not find the directory: {raw_dir}. Make sure it exists and contains subdirectories for classes.")
    except Exception as e:
        logging.error(f"An error occurred during preprocessing: {e}")

if __name__ == "__main__":
    preprocess_images(RAW_DATA_DIR, PROCESSED_DATA_DIR, transform)