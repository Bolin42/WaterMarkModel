import torch
import torch.nn as nn
from config import config

class WatermarkEncoder(nn.Module):
    """
    A neural network to encode the watermark into a latent space embedding.
    """
    def __init__(self):
        super(WatermarkEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(config.WATERMARK_LENGTH, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, config.LATENT_DIM * (config.IMG_SIZE // 8) * (config.IMG_SIZE // 8))
        )

    def forward(self, watermark):
        """
        Args:
            watermark (torch.Tensor): A tensor of shape (batch_size, watermark_length).

        Returns:
            torch.Tensor: The watermark embedding in the latent space,
                          reshaped to (batch_size, latent_dim, height, width).
        """
        embedding = self.encoder(watermark)
        return embedding.view(
            -1,
            config.LATENT_DIM,
            config.IMG_SIZE // 8,
            config.IMG_SIZE // 8
        )

if __name__ == '__main__':
    # A simple test case for the encoder
    encoder = WatermarkEncoder().to(config.DEVICE)
    watermark = torch.randn(2, config.WATERMARK_LENGTH).to(config.DEVICE)
    embedding = encoder(watermark)
    print(f"Encoder output shape: {embedding.shape}")
    assert embedding.shape == (2, config.LATENT_DIM, config.IMG_SIZE // 8, config.IMG_SIZE // 8)
    print("WatermarkEncoder test passed.")
