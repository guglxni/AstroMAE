# This script will handle downloading SDSS Galaxy10 data

import os
import numpy as np
from astroNN.datasets import load_galaxy10
from astroNN.datasets.galaxy10 import galaxy10cls_lookup
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the directory to save raw images
RAW_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'raw'))
NUM_IMAGES_TO_DOWNLOAD = 1000

def download_and_save_galaxy10(num_images, save_dir):
    """Downloads Galaxy10 dataset and saves images to specified directory."""
    logging.info("Starting Galaxy10 dataset download...")
    try:
        # astroNN downloads and caches the data automatically if not found
        # It returns images and labels
        images, labels = load_galaxy10()
        logging.info(f"Loaded {len(images)} images from Galaxy10 dataset.")

        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)
        logging.info(f"Ensured raw data directory exists: {save_dir}")

        # Limit the number of images if requested
        if num_images > len(images):
            logging.warning(f"Requested {num_images} images, but dataset only contains {len(images)}. Using all available images.")
            num_images = len(images)
        else:
            logging.info(f"Selecting the first {num_images} images.")

        images = images[:num_images]
        labels = labels[:num_images]

        # Save images to the directory, organized by class
        saved_count = 0
        for i, (img_array, label_idx) in enumerate(zip(images, labels)):
            try:
                # Get class name from label index
                class_name = galaxy10cls_lookup(label_idx)
                class_dir = os.path.join(save_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)

                # Convert numpy array (H, W, C) to PIL Image
                # Ensure pixel values are in the range [0, 255]
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)

                img = Image.fromarray(img_array)

                # Save image as PNG
                img_filename = f"galaxy_{i:04d}.png"
                img_path = os.path.join(class_dir, img_filename)
                img.save(img_path)
                saved_count += 1
            except Exception as e:
                logging.error(f"Failed to process or save image {i}: {e}")

        logging.info(f"Successfully saved {saved_count} images to {save_dir}")

    except ImportError:
        logging.error("astroNN library not found. Please install it: pip install astroNN")
    except Exception as e:
        logging.error(f"An error occurred during download or saving: {e}")

if __name__ == "__main__":
    download_and_save_galaxy10(NUM_IMAGES_TO_DOWNLOAD, RAW_DATA_DIR)