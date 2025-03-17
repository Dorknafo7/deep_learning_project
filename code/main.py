import torch
from torchvision import datasets, transforms
import numpy as np
from matplotlib import pyplot as plt
from utils import plot_tsne
import utils
import numpy as np
import random
import argparse
import AutoEncoderDecoder
from torch.utils.data import Dataset, DataLoader, random_split

NUM_CLASSES = 10

def freeze_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
def get_args():   
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators')
    parser.add_argument('--data-path', default="/datasets/cv_datasets/data", type=str, help='Path to dataset')
    parser.add_argument('--batch-size', default=8, type=int, help='Size of each batch')
    parser.add_argument('--latent-dim', default=128, type=int, help='encoding dimension')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Default device to use')
    parser.add_argument('--mnist', action='store_true', default=False,
                        help='Whether to use MNIST (True) or CIFAR10 (False) data')
    parser.add_argument('--self-supervised', action='store_true', default=False,
                        help='Whether train self-supervised with reconstruction objective, or jointly with classifier for classification objective.')
    return parser.parse_args()
    

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    args = get_args()
    freeze_seeds(args.seed)
                                      
    # if args.mnist:
       # train_dataset = datasets.MNIST(root=args.data_path, train=True, download=False, transform=transform)
        #test_dataset = datasets.MNIST(root=args.data_path, train=False, download=False, transform=transform)
    # else:
    #     train_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
    #     test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform)
        
    # Split training data into training and validation sets, 80% for training, 20% for validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create DataLoader 
    dl_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dl_test  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    dl_val   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    im_size = train_dataset[0][0].shape
                                           
    encoder = AutoEncoderMnist(in_channels=im_size[0] ,out_channels=args.latent_dim)
    decoder = AutoDecoderMnist(in_channels=args.latent_dim, out_channels=im_size[0])

    epochs = 100
    train(encoder, decoder, epochs, device)

    batch_images, _ = next(iter(dl_train))  # Get one batch of images
    batch_images = batch_images.to(device)  # Move to GPU if available
    encoded = encoder(batch_images)
    decoded = decoder(encoded)  
    print(decoded.shape)

