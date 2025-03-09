import torch
from torchvision import datasets, transforms
import numpy as np
import random
import argparse
from matplotlib import pyplot as plt
from utils import plot_tsne

NUM_CLASSES = 10

# Function to set the seed for reproducibility
def freeze_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# Argument parser for command-line arguments
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators')
    parser.add_argument('--data-path', default="~/datasets/cv_datasets/data", type=str, help='Path to dataset')
    parser.add_argument('--batch-size', default=8, type=int, help='Size of each batch')
    parser.add_argument('--latent-dim', default=128, type=int, help='Encoding dimension')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Default device to use')
    parser.add_argument('--mnist', action='store_true', default=False, help='Use MNIST (True) or CIFAR10 (False) data')
    parser.add_argument('--self-supervised', action='store_true', default=False, help='Train self-supervised or jointly with classifier')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debugging for dataloader')
    return parser.parse_args()

# Autoencoder definition
class Autoencoder(torch.nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
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

# Classifier definition (for supervised training)
class Classifier(torch.nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = torch.nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# Function to rescale images back from [-1, 1] to [0, 1]
def rescale_image(image):
    return np.clip((image + 1) / 2, 0, 1)  # Rescale from [-1, 1] to [0, 1]

# Function to plot reconstructed images (rescale after reconstruction)
def plot_reconstruction(original, reconstructed, num_images=10):
    original = original.cpu().detach().numpy()
    reconstructed = reconstructed.cpu().detach().numpy()
    
    # Rescale images to [0, 1] range for visualization
    original = rescale_image(original)
    reconstructed = rescale_image(reconstructed)
    num_images = min(num_images, original.shape[0])  # Ensure we don't exceed batch size
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 20))

    for i in range(num_images):
        ax = axes[i, 0]
        ax.imshow(np.transpose(original[i], (1, 2, 0)))  # Reorder dimensions to (H, W, C)
        ax.set_title('Original')
        ax.axis('off')

        ax = axes[i, 1]
        ax.imshow(np.transpose(reconstructed[i], (1, 2, 0)))
        ax.set_title('Reconstructed')
        ax.axis('off')

    plt.show()
# Helper function to unnormalize images from the transform (which uses mean=0.5, std=0.5)
def unnormalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    # Assume img shape is (B, C, H, W)
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    return img * std + mean

# Main function for training and evaluation
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalization for CIFAR-10
    ])

    # Parse arguments
    args = get_args()
    freeze_seeds(args.seed)

    # Load the dataset (CIFAR-10 or MNIST)
    if args.mnist:
        train_dataset = datasets.MNIST(root=args.data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=args.data_path, train=False, download=True, transform=transform)
    else:
        train_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform)

    # Data loaders for training and testing
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize the autoencoder and the optimizer
    latent_dim = args.latent_dim
    autoencoder = Autoencoder(latent_dim).to(args.device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)

    # Training loop for self-supervised task
    num_epochs = 10
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for data in train_loader:
            images, _ = data
            images = images.to(args.device)
            optimizer.zero_grad()
            reconstructed, _ = autoencoder(images)
            loss = criterion(reconstructed, images)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)  # Average loss for this epoch
        print(f"Epoch {epoch+1}, Loss: {avg_epoch_loss}")

    # Plot reconstructions after training
    with torch.no_grad():
        test_images, _ = next(iter(test_loader))
        test_images = test_images.to(args.device)
        reconstructed, _ = autoencoder(test_images)
        plot_reconstruction(test_images, reconstructed)

    # Train classifier using the pre-trained encoder
    for param in autoencoder.encoder.parameters():
       param.requires_grad = False  # Freeze encoder weights

    classifier = Classifier(latent_dim, NUM_CLASSES).to(args.device)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    # Training loop for classifier:
    # num_epochs = 10
    # for epoch in range(num_epochs):
    #     for data in train_loader:
    
    #         images, labels = data
    #         images = images.to(args.device)
    #         labels = labels.to(args.device)

    #         with torch.no_grad():
    #             latent = autoencoder.encoder(images)

    #         classifier_optimizer.zero_grad()
    #         outputs = classifier(latent)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         classifier_optimizer.step()

    #     print(f"Epoch {epoch+1}, Classifier Loss: {loss.item()}")

    # # Visualize latent space
    # plot_tsne(autoencoder.encoder, test_loader, args.device)
