import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoderCIFAR(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32*32*3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, latent_dim)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32*32*3),
            torch.nn.Tanh()  # For reconstruction
        )

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        x_reconstructed = x_reconstructed.view(-1, 3, 32, 32)
        return x_reconstructed, z

class AutoEncoderMnist(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Calculate the output size after convolutions
        # input size is 28x28x1 (MNIST grayscale)
        self.flattened_size = self._get_flattened_size(in_channels)

        # Fully connected layer (latent space)
        self.fc1 = nn.Linear(self.flattened_size, latent_dim)

    def _get_flattened_size(self, in_channels):
        # Create a dummy tensor to calculate the output size after convolutions
        dummy_input = torch.zeros(1, in_channels, 28, 28)
        dummy_output = self.cnn(dummy_input)
        return int(torch.prod(torch.tensor(dummy_output.shape[1:])))

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.shape[0], -1)  # Flatten the output while preserving batch size
        x = self.fc1(x)
        return x


class AutoDecoderMnist(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),  # Latent vector to image features
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),  # Reshape to 128 channels, 3x3 feature maps
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.Tanh()  # Scale output to [-1, 1]
        )

    def forward(self, h):
        x = self.decoder(h)
        return x


def reconstruction_loss(x, x_rec):
    return F.mse_loss(x_rec, x)  # Using MSE loss for reconstruction error

def train(encoder, decoder, epochs, dl_train, device):

    # Hyper-parameters?

    # Optimizer
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.01)

    for epoch in range(epochs):
        total_train_loss = 0.0
        num_train_samples = 0

        encoder.train()
        decoder.train()

        i = 0

        # Iterate over training data
        for images, _ in dl_train:
            print(i)
            images = images.to(device)  # Move to GPU if available
            
            optimizer.zero_grad()  # Zero the gradients

            # Forward pass: Encode and then Decode
            encoded = encoder(images)
            decoded = decoder(encoded)

            # Calculate the reconstruction loss
            loss = reconstruction_loss(images, decoded)  # MSE loss between reconstructed and original images
            total_train_loss += loss.item()
            num_train_samples += images.size(0)  # Count number of samples in this batch

            # Backpropagation
            loss.backward()
            optimizer.step()

            i = i + 1

        # Calculate average loss for this epoch
        avg_train_loss = total_train_loss / num_train_samples
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}")



