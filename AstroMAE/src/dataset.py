import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob

class AstroDataset(Dataset):
    def __init__(self, root_dir, image_size=128):
        self.root_dir = root_dir
        # Assuming images are directly in the root_dir (e.g., .png, .jpg, .fits)
        # Adjust the pattern if images are in subfolders or have different extensions
        self.image_paths = glob.glob(os.path.join(root_dir, '*.[jp][pn]g')) + \
                           glob.glob(os.path.join(root_dir, '*.fits')) # Add other formats if needed

        if not self.image_paths:
            print(f"Warning: No images found in {root_dir} with patterns '*.[jp][pn]g' or '*.fits'")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3), # Convert to 3 channels if needed by ViT
            transforms.ToTensor(),
            # Add normalization if required, e.g.:
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            # Handle FITS files if necessary, requires astropy or similar
            if img_path.lower().endswith('.fits'):
                # Placeholder: Add FITS loading logic here if needed
                # from astropy.io import fits
                # with fits.open(img_path) as hdul:
                #     image = Image.fromarray(hdul[0].data.astype('float32'), mode='F') # Example
                # For now, raise error if FITS encountered without specific loader
                raise NotImplementedError(f"FITS file loading not implemented for {img_path}")
            else:
                image = Image.open(img_path).convert('RGB') # Ensure RGB for standard transforms

            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy tensor or skip this sample
            return torch.zeros((3, self.transform.transforms[0].size[0], self.transform.transforms[0].size[1]))

def create_dataloader(dataset_dir, image_size, batch_size, num_workers=0):
    dataset = AstroDataset(root_dir=dataset_dir, image_size=image_size)
    if len(dataset) == 0:
        print(f"Error: Dataset at {dataset_dir} is empty or couldn't find images.")
        return None
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader