"""Model definitions for image classification.

Provides a small convolutional network suitable for 64x64 RGB inputs and a
binary classification head by default. The architecture is intentionally simple
and fast to train for experiments and educational purposes.
"""

from torch import nn


class SimpleCNN(nn.Module):
    """A compact CNN for small RGB images.

    The network consists of three convolutional blocks with ReLU activations
    and downsampling, followed by global average pooling and a small MLP
    classifier. Designed for inputs around 64x64 pixels.

    Args:
        num_classes: Number of output classes for the classifier head.
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=False),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """Compute logits for input batch ``x``.

        Args:
            x: Tensor of shape ``(N, 3, H, W)`` with RGB images.

        Returns:
            Tensor of shape ``(N, num_classes)`` with unnormalized logits.
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


