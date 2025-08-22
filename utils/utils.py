import logging
import os
import torch
from PIL import Image
import numpy as np
from config import config

def setup_logging():
    """Sets up the logging for the project."""
    os.makedirs(config.LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE, mode='a'),
            logging.StreamHandler()
        ]
    )

def save_checkpoint(epoch, model, optimizer, path):
    """Saves a model checkpoint."""
    logging.info(f"Saving checkpoint to {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def load_checkpoint(model, optimizer, path):
    """Loads a model checkpoint."""
    if os.path.exists(path):
        logging.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        return epoch
    else:
        logging.warning(f"Checkpoint not found at {path}. Starting from scratch.")
        return 0

def save_image(tensor, path):
    """Saves a tensor as an image."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Ensure tensor is in the correct format (e.g., float, not half)
    tensor = tensor.float().detach()

    # Permute if necessary from (C, H, W) to (H, W, C)
    if tensor.dim() == 3 and tensor.shape[0] in [1, 3, 4]:
        tensor = tensor.permute(1, 2, 0)

    # Squeeze singleton dimensions
    tensor = tensor.squeeze()

    # Clamp and scale to 0-255
    image_np = (tensor.clamp(0, 1) * 255).numpy().astype(np.uint8)

    # Handle grayscale images
    if image_np.ndim == 2:
        image = Image.fromarray(image_np, 'L')
    else:
        image = Image.fromarray(image_np)
        
    image.save(path)
    logging.info(f"Saved image to {path}")
