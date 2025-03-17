import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # Define the layers
        self.fc1 = torch.nn.Linear(latent_dim, 512)
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(512)  # Batch normalization after the first hidden layer
        self.drop1 = torch.nn.Dropout(0.5)  # Dropout layer to prevent overfitting

        self.fc2 = torch.nn.Linear(512, 256)
        self.relu2 = torch.nn.ReLU()
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.drop2 = torch.nn.Dropout(0.5)

        self.fc3 = torch.nn.Linear(256, num_classes)  # Final output layer (classification)

    def forward(self, x):
        x = self.relu1(self.bn1(self.fc1(x)))  # Apply fc1 + ReLU + BatchNorm
        x = self.drop1(x)  # Apply dropout
        
        x = self.relu2(self.bn2(self.fc2(x)))  # Apply fc2 + ReLU + BatchNorm
        x = self.drop2(x)  # Apply dropout
        
        x = self.fc3(x)  # Final classification layer (logits)
        return x
    
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
    return F.l1_loss(x_rec, x)  # Using L1 loss for Mean Absolute Error (MAE)

def trainEncoder(encoder, decoder, epochs, dl_train, dl_val, device):
    print("Train AutoEncoder")
    # Optimizer
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.01)
    
    for epoch in range(epochs):
        total_train_loss = 0.0
        num_train_samples = 0

        encoder.train()  # Set encoder to training mode
        decoder.train()  # Set decoder to training mode

        # Iterate over training data
        for images, _ in dl_train:
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

        # Calculate average loss for this epoch
        avg_train_loss = total_train_loss / num_train_samples

        # Validation phase (no gradients needed)
        encoder.eval()  # Set encoder to evaluation mode
        decoder.eval()  # Set decoder to evaluation mode

        total_val_loss = 0.0
        num_val_samples = 0

        with torch.no_grad():  # Disable gradient computation
            for images, _ in dl_val:
                images = images.to(device)

                # Forward pass: Encode and then Decode
                encoded = encoder(images)
                decoded = decoder(encoded)

                # Calculate the reconstruction loss for validation
                loss = reconstruction_loss(images, decoded)
                total_val_loss += loss.item()
                num_val_samples += images.size(0)  # Count number of samples in this batch

        # Calculate average validation loss for this epoch
        avg_val_loss = total_val_loss / num_val_samples

        # Print stats for this epoch
        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Validation Loss: {avg_val_loss:.4f}")

        # Set the model back to training mode for the next epoch
        encoder.train()
        decoder.train()


class Classifier(nn.Module):
    def __init__(self, latent_dim=128):
        super(Classifier, self).__init__()
        # A simple fully connected layer to classify
        self.fc = nn.Linear(latent_dim, 10)  # 10 classes for MNIST

    def forward(self, x):
        return self.fc(x)

def trainClassifier(encoder, classifier, epochs, dl_train, dl_val, device):
    print("Train Classifier")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    for epoch in range(epochs):
        classifier.train()  # Set classifier to training mode
        classifier.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dl_train:
            images = images.to(device)  # Move to GPU if available
            labels = labels.to(device)  # Move labels to GPU
            with torch.no_grad():  # Don't compute gradients for encoder
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
        classifier.eval()  # Set classifier to evaluation mode
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():  # No gradient computation during validation
            for images, labels in dl_val:
                images = images.to(device)  # Move to GPU if available
                labels = labels.to(device)  # Move labels to GPU
                latent_vectors = encoder(images)
                outputs = classifier(latent_vectors)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Print training and validation stats
        print(f"Epoch [{epoch+1}/{epochs}], "
            f"Train Loss: {running_loss/len(dl_train):.4f}, "
            f"Train Accuracy: {100 * correct/total:.2f}%, "
            f"Validation Loss: {val_loss/len(dl_val):.4f}, "
            f"Validation Accuracy: {100 * val_correct/val_total:.2f}%")

def evaluateClassifier(encoder, classifier, dl_test, device):
    print("Evaluate Classifier")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    classifier.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in dl_test:
            images = images.to(device)  # Move to GPU if available
            latent_vectors = encoder(images)
            outputs = classifier(latent_vectors)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    print(f"Test Loss: {test_loss/len(dl_test):.4f}, "
        f"Test Accuracy: {100 * test_correct/test_total:.2f}%")


####################### Section 1_2_2 #######################

class ClassifierModel_122(nn.Module):
    def __init__(self, encoder, classifier):
        super(ClassifierModel_122, self).__init__()
        self.encoder = encoder  # Encoder (AutoEncoder part)
        self.classifier = classifier  # Classifier (FC layer to classify the latent vector)

    def forward(self, x):
        latent_vector = self.encoder(x)  # Pass input through the encoder
        output = self.classifier(latent_vector)  # Classify based on the latent vector
        return output

def trainClassifier122(model, epochs, dl_train, dl_val, device):
    print("Train Encoder + Classifier")
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Adam optimizer for both encoder and classifier

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dl_train:
            images = images.to(device)  # Move to GPU if available
            labels = labels.to(device)  # Move labels to GPU

            # Forward pass: Get classification output
            outputs = model(images)  # This uses the encoder and classifier

            # Compute loss
            loss = criterion(outputs, labels)

            # Backpropagation and optimization
            optimizer.zero_grad()  # Zero the gradients
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update the weights

            # Update statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Validation phase (no gradients needed)
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in dl_val:
                images = images.to(device)  # Move to GPU if available
                labels = labels.to(device)  # Move labels to GPU
                
                outputs = model(images)  # Get classification output from model
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Print training and validation stats
        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {running_loss/len(dl_train):.4f}, "
              f"Train Accuracy: {100 * correct/total:.2f}%, "
              f"Validation Loss: {val_loss/len(dl_val):.4f}, "
              f"Validation Accuracy: {100 * val_correct/val_total:.2f}%")


def evaluateClassifier122(model, dl_test, device):
    model.eval()  # Set model to evaluation mode
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification

    total_test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for images, labels in dl_test:
            images = images.to(device)  # Move to GPU if available
            labels = labels.to(device)  # Move labels to GPU

            # Forward pass: Get classification output
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


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, z_i, z_j):
        # Normalize the embeddings
        z_i = F.normalize(z_i, dim=-1, p=2)
        z_j = F.normalize(z_j, dim=-1, p=2)
        
        # Concatenate both views
        representations = torch.cat([z_i, z_j], dim=0)
        
        # Similarity matrix
        similarity_matrix = torch.matmul(representations, representations.T)
        
        # Mask out the diagonal (same image pairs)
        batch_size = z_i.size(0)
        mask = torch.eye(batch_size * 2).bool().to(z_i.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # Apply temperature scaling
        similarity_matrix /= self.temperature
        
        # Compute the contrastive loss
        labels = torch.cat([torch.arange(batch_size).to(z_i.device), 
                            torch.arange(batch_size).to(z_i.device)], dim=0)
        
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss


def trainEncoder123(model, epochs, dl_train, device):
    print("trainEncoder123")
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Loss function
    criterion = NTXentLoss(temperature=0.5)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, _ in dl_train:
            images = images.to(device)

            # Create augmented views (e.g., flipping as augmentation)
            x_i = images
            x_j = torch.flip(x_i, dims=[3])  # Example: flip as another augmentation

            # Forward pass
            z_i = model(x_i)
            z_j = model(x_j)

            # Compute contrastive loss
            loss = criterion(z_i, z_j)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dl_train)}")

