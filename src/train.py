import torch
import yaml
import os
import argparse
import pandas as pd
from dataset import create_dataloader
from model import MaskedAutoencoder
import torch.optim as optim

# Training loop
class Trainer:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        print(f"Using device: {self.device}")

        # Create necessary directories
        self.checkpoint_dir = self.config['checkpoint_dir'] # Relative to project root
        self.log_dir = 'outputs' # Relative to project root
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, 'train_log.csv')

        # Data Loader
        dataset_path = self.config['dataset_dir'] # Relative to project root
        self.dataloader = create_dataloader(
            dataset_dir=dataset_path,
            image_size=self.config['image_size'],
            batch_size=self.config['batch_size']
        )
        if self.dataloader is None:
            raise ValueError("Failed to create dataloader. Check dataset path and contents.")

        # Model
        # Assuming default ViT base parameters for now, adjust if needed
        self.model = MaskedAutoencoder(
            img_size=self.config['image_size'],
            patch_size=16, # Standard for ViT-Base
            in_chans=3,
            embed_dim=768, # ViT-Base
            depth=12,      # ViT-Base
            num_heads=12,    # ViT-Base
            decoder_embed_dim=512,
            decoder_depth=8,
            decoder_num_heads=16,
            mlp_ratio=4.0,
            norm_layer=torch.nn.LayerNorm
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'], weight_decay=0.05)

        # Logging setup
        self.log_data = []

    def train(self):
        print("Starting training...")
        self.model.train()
        for epoch in range(self.config['epochs']):
            epoch_loss = 0.0
            num_batches = 0
            for batch_idx, inputs in enumerate(self.dataloader):
                inputs = inputs.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass (model handles masking and loss calculation)
                loss, _, _ = self.model(inputs, mask_ratio=self.config['mask_ratio'])

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                if batch_idx % 50 == 0: # Log progress within epoch
                    print(f'  Epoch [{epoch+1}/{self.config["epochs"]}], Batch [{batch_idx+1}/{len(self.dataloader)}], Loss: {loss.item():.4f}')

            avg_epoch_loss = epoch_loss / num_batches
            print(f'Epoch [{epoch+1}/{self.config["epochs"]}], Average Loss: {avg_epoch_loss:.4f}')

            # Log epoch loss
            self.log_data.append({'epoch': epoch + 1, 'loss': avg_epoch_loss})
            pd.DataFrame(self.log_data).to_csv(self.log_file, index=False)

            # Save checkpoint
            if (epoch + 1) % 5 == 0 or (epoch + 1) == self.config['epochs']:
                checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_epoch_loss,
                    'config': self.config
                }, checkpoint_path)
                print(f'Checkpoint saved to {checkpoint_path}')

        print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train AstroMAE Model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    args = parser.parse_args()

    try:
        trainer = Trainer(config_path=args.config)
        trainer.train()
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
    except Exception as e:
        print(f"An error occurred during training: {e}")