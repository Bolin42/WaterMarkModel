import torch
import torch.nn as nn
from config import config

class WatermarkDecoder(nn.Module):
    """
    A neural network to decode the watermark from a latent space representation.
    """
    def __init__(self):
        super(WatermarkDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(config.LATENT_DIM, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, config.WATERMARK_LENGTH)
        )

    def forward(self, latent_representation):
        """
        Args:
            latent_representation (torch.Tensor): The latent representation of the
                                                  watermarked image, shape
                                                  (batch_size, latent_dim, height, width).

        Returns:
            torch.Tensor: The recovered watermark, shape (batch_size, watermark_length).
        """
        return self.decoder(latent_representation)

if __name__ == '__main__':
    # A simple test case for the decoder
    decoder = WatermarkDecoder().to(config.DEVICE)
    latent = torch.randn(2, config.LATENT_DIM, config.IMG_SIZE // 8, config.IMG_SIZE // 8).to(config.DEVICE)
    watermark = decoder(latent)
    print(f"Decoder output shape: {watermark.shape}")
    assert watermark.shape == (2, config.WATERMARK_LENGTH)
    print("WatermarkDecoder test passed.")
