import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import  transforms
import torchvision.transforms.functional as TF
import numpy as np
from matplotlib import pyplot as plt

##################### CIFAR 1_2_1 #####################

class EncoderCIFAR(nn.Module):
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
        self.flattened_size = 128 * 4 * 4  
        self.fc1 = nn.Linear(self.flattened_size, latent_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.shape[0], -1)  
        x = self.fc1(x)
        return x

class DecoderCIFAR(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),  
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),          

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  
        )

    def forward(self, h):
        x = self.decoder(h)
        return x

class ClassifierCIFAR(torch.nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(ClassifierCIFAR, self).__init__()

        self.fc1 = torch.nn.Linear(latent_dim, 512)
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(512)  
        self.drop1 = torch.nn.Dropout(0.5)  

        self.fc2 = torch.nn.Linear(512, 256)
        self.relu2 = torch.nn.ReLU()
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.drop2 = torch.nn.Dropout(0.5)

        self.fc3 = torch.nn.Linear(256, num_classes) 

    def forward(self, x):
        x = self.relu1(self.bn1(self.fc1(x)))  
        x = self.drop1(x)
        
        x = self.relu2(self.bn2(self.fc2(x)))  
        x = self.drop2(x)  
        
        x = self.fc3(x) 
        return x

def rescale_image(image):
    return np.clip((image + 1) / 2, 0, 1)  # Rescale from [-1, 1] to [0, 1]

def plot_reconstruction(original, reconstructed, num_images=5):
    original = original.cpu().detach().numpy()
    reconstructed = reconstructed.cpu().detach().numpy()
    
    original = rescale_image(original)
    reconstructed = rescale_image(reconstructed)
    num_images = min(num_images, original.shape[0])  # Ensure we don't exceed batch size
    fig, axes = plt.subplots(num_images, 2, figsize=(num_images * 2, 4))

    for i in range(num_images):
        ax = axes[0, i]
        ax.imshow(np.transpose(original[i], (1, 2, 0)))  # Reorder dimensions to (H, W, C)
        ax.set_title('Original')
        ax.axis('off')

        ax = axes[1, i]
        ax.imshow(np.transpose(reconstructed[i], (1, 2, 0)))
        ax.set_title('Reconstructed')
        ax.axis('off')

    plt.show()

def plot_images_with_labels(images, true_labels, predicted_labels, class_names, num_images=10):
    images = images.cpu().detach().numpy()  
    true_labels = true_labels.cpu().detach().numpy()  
    predicted_labels = predicted_labels.cpu().detach().numpy()  
    
    num_images = min(num_images, images.shape[0]) 
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))

    for i in range(num_images):
        ax = axes[i]
        ax.imshow(np.transpose(images[i], (1, 2, 0)))  # Reorder to (H, W, C)
        ax.set_title(f"True: {class_names[true_labels[i]]}\nPred: {class_names[predicted_labels[i]]}")
        ax.axis('off')  
    plt.show()   

##################### MNIST 1_2_1 #####################
    
class EncoderMnist(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super().__init__()

        self.flattened_size = in_channels * 28 * 28  # MNIST images are 28x28
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc_layers(x)
        return x


class DecoderMnist(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super().__init__()

        self.out_channels = out_channels

        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, out_channels * 28 * 28),
            nn.Tanh()
        )

    def forward(self, h):
        x = self.fc_layers(h)
        x = x.view(-1, self.out_channels, 28, 28)
        return x

def reconstruction_loss(x, x_rec):
    return F.l1_loss(x_rec, x)  # Using L1 loss for Mean Absolute Error (MAE)

def trainEncoderMNIST(encoder, decoder, epochs, dl_train, dl_val, device):
    print("Train Encoder")
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.0001)
    
    for epoch in range(epochs):
        total_train_loss = 0.0
        num_train_samples = 0

        encoder.train()
        decoder.train()

        # Iterate over training data
        for images, _ in dl_train:
            images = images.to(device)
            batch_size = images.size(0)

            optimizer.zero_grad()

            # Forward pass
            encoded = encoder(images)
            decoded = decoder(encoded)

            # Calculate the reconstruction loss
            loss = reconstruction_loss(images, decoded)
            total_train_loss += loss.item()
            num_train_samples += batch_size

            # Backpropagation
            loss.backward()
            optimizer.step()

        # Calculate average loss for this epoch
        avg_train_loss = total_train_loss / num_train_samples

        # Validation phase (no gradients needed)
        encoder.eval()
        decoder.eval()

        total_val_loss = 0.0
        num_val_samples = 0

        with torch.no_grad():
            for images, _ in dl_val:
                images = images.to(device)

                encoded = encoder(images)
                decoded = decoder(encoded)

                loss = reconstruction_loss(images, decoded)
                total_val_loss += loss.item()
                num_val_samples += images.size(0)

        avg_val_loss = total_val_loss / num_val_samples

        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Validation Loss: {avg_val_loss:.4f}")

        # Set the model back to training mode for the next epoch
        encoder.train()
        decoder.train()


class ClassifierMNIST(nn.Module):
    def __init__(self, latent_dim=128):
        super(ClassifierMNIST, self).__init__()
        # A simple fully connected layer to classify
        self.fc = nn.Linear(latent_dim, 10)  # 10 classes for MNIST

    def forward(self, x):
        return self.fc(x)

def trainClassifierMNIST(encoder, classifier, epochs, dl_train, dl_val, device):
    print("Train Classifier")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    for epoch in range(epochs):
        classifier.train()
        classifier.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dl_train:
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                latent_vectors = encoder(images)
            
            outputs = classifier(latent_vectors)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Validation phase (no gradients needed)
        classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in dl_val:
                images = images.to(device)
                labels = labels.to(device)
                latent_vectors = encoder(images)
                outputs = classifier(latent_vectors)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        print(f"Epoch [{epoch+1}/{epochs}], "
            f"Train Loss: {running_loss/len(dl_train):.4f}, "
            f"Train Accuracy: {100 * correct/total:.2f}%, "
            f"Validation Loss: {val_loss/len(dl_val):.4f}, "
            f"Validation Accuracy: {100 * val_correct/val_total:.2f}%")

def evaluateClassifierMNIST(encoder, classifier, dl_test, device):
    print("Evaluate Classifier")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    classifier.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in dl_test:
            images = images.to(device)
            latent_vectors = encoder(images)
            outputs = classifier(latent_vectors)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    print(f"Test Loss: {test_loss/len(dl_test):.4f}, "
        f"Test Accuracy: {100 * test_correct/test_total:.2f}%")


####################### MNIST 1_2_2 #######################

class ClassifierMNIST122(nn.Module):
    def __init__(self, encoder, classifier):
        super(ClassifierMNIST122, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        latent_vector = self.encoder(x)
        output = self.classifier(latent_vector)
        return output

def trainClassifierMNIST122(model, epochs, dl_train, dl_val, device):
    print("Train Encoder + Classifier")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dl_train:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Validation phase (no gradients needed)
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in dl_val:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {running_loss/len(dl_train):.4f}, "
              f"Train Accuracy: {100 * correct/total:.2f}%, "
              f"Validation Loss: {val_loss/len(dl_val):.4f}, "
              f"Validation Accuracy: {100 * val_correct/val_total:.2f}%")


def evaluateClassifierMNIST122(model, dl_test, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dl_test:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate classification loss
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate average test loss
    avg_test_loss = total_test_loss / len(dl_test)
    test_accuracy = 100 * correct / total

    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    return avg_test_loss, test_accuracy


####################### Section 1_2_3 #######################

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=128, out_dim=128):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(512, out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: Tensor of shape (batch_size, feature_dim)
        labels: Tensor of shape (batch_size)
        """
        features = F.normalize(features, dim=-1, p=2)
        batch_size = features.shape[0]

        similarity_matrix = torch.exp(torch.mm(features, features.T) / self.temperature)

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)

        mask.fill_diagonal_(0)

        # Compute contrastive loss
        pos_sim = (similarity_matrix * mask).sum(dim=1)
        neg_sim = similarity_matrix.sum(dim=1)

        loss = -torch.log(pos_sim / neg_sim).mean()
        return loss

def trainEncoderMNIST123(model, epochs, dl_train, device):
    print("Training Encoder with SupConLoss")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = SupConLoss(temperature=0.1)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, labels in dl_train:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            features = model(images)

            # Compute supervised contrastive loss
            loss = criterion(features, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dl_train)}")
    
def train_encoder_cifar(model, projection_head, epochs, dl_train, device):
    print("Training 1.2.3 contrastive encoder for CIFAR")
    optimizer = torch.optim.Adam(list(model.parameters()) + list(projection_head.parameters()), lr=1e-3)
    criterion = SupConLoss(temperature=0.5)
    # Training loop
    for epoch in range(epochs):
        model.train()
        projection_head.train()
        total_loss = 0.0

        for images, _ in dl_train:
            images = images.to(device)
            
            x_i = images
            x_j = torch.flip(x_i, dims=[3])
            x_i, x_j = x_i.to(device), x_j.to(device)

            # Encode images
            z_i = model(x_i)
            z_j = model(x_j)
        
            # Compute contrastive loss
            loss = criterion(z_i, z_j)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dl_train)}")
