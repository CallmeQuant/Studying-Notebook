import torch
import torch.nn as nn
import torch.optim as optim # optimization algo
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Transform to tensor
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# MNIST data
train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('data', train=False, download=True, transform=transform)

# train test split
train_size = int(0.8 * len(train_data))
valid_size = len(train_data) - train_size
train_dataset, validation_dataset = torch.utils.data.random_split(train_data, [train_size, valid_size])

# DataLoader for train, validation, test set
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

# for test_images, test_labels in test_loader:
#     sample_image = test_images[0]    # Reshape them according to your needs.
#     sample_label = test_labels[0]
#     print(sample_image.shape)
#     print(sample_label.shape)
# Define neural network

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize neural network
model = NeuralNet()

# Loss function and optimizer
lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Placeholder for training and validation loss
train_losses = []
validation_losses = []

train_acc_ = []
val_acc_ = []
# Number of epoches
num_epoch = 10
# Check if cuda is available

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

if use_cuda:
      model = model.cuda()
      criterion = criterion.cuda()

for epoch in range(1, num_epoch+1):
    # Initialize training mode
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    total_train_samples = 0.0

    optimizer.zero_grad()
    for data, test_label in tqdm(train_loader):
        batch_size = test_label.size(0)
        total_train_samples += batch_size
        test_label = test_label.to(device)
        data = data.to(device)

        output = model(data)
        # Compute loss
        loss = criterion(output, test_label)
        # clear gradient
        optimizer.zero_grad()
        # Backward pass: compute gradient of the loss with
        # respect to all the learnable parameters
        loss.backward()
        optimizer.step()
        # Compute acc
        acc = (output.argmax(dim=1) == test_label).sum().item()
        train_acc += acc
        train_loss += loss.item()

    loss_at_epoch = train_loss / len(train_loader)
    acc_at_epoch = train_acc / total_train_samples

    train_losses.append(loss_at_epoch)
    train_acc_.append(acc_at_epoch)


    # Evaluation stage
    model.eval()
    validation_loss = 0.0
    validation_acc = 0.0
    total_validation_samples = 0.0
    with torch.no_grad():
        for val_data, val_label in validation_loader:
            batch_size = val_label.size(0)
            total_validation_samples += batch_size
            val_label = val_label.to(device)
            val_data = val_data.to(device)

            output = model(val_data)
            # Compute loss
            loss = criterion(output, val_label)
            # Compute acc
            acc = (output.argmax(dim=1) == val_label).sum().item()
            validation_acc += acc
            validation_loss += loss.item()

        loss_validation_at_epoch = validation_loss / len(validation_loader)
        validation_losses.append(loss_validation_at_epoch)
        acc_validation_at_epoch = validation_acc / total_validation_samples
        val_acc_.append(acc_validation_at_epoch)

    if (int(epoch) % 2) == 0:
        print(
        f'Epoch: {epoch:>3}'
        + f' | Train loss: {loss_at_epoch:.3f}'
        + f' | Train acc: {acc_at_epoch:.3f}'
        + f' | Validation loss: {loss_validation_at_epoch:.3f}'
        + f' | Validation acc: {acc_validation_at_epoch:.3f}')

# Test set
model.eval()
test_loss = 0.0
test_acc = 0.0
predictions = []
targets = []

for data, test_label in tqdm(test_loader):
    data = data.to(device)
    test_label = test_label.to(device)
    output = model(data)
    loss = criterion(output, test_label)
    test_loss += loss.item()
    acc = (output.argmax(dim=1) == test_label).sum().item()
    test_acc += acc
    _, predicted = torch.max(output, 1)
    predictions.extend(predicted.detach().cpu().numpy())
    targets.extend(test_label.numpy())


from sklearn.metrics import accuracy_score
from sklearn.metrics import  classification_report, confusion_matrix
accuracy = accuracy_score(targets, predictions)
confusion_mat = confusion_matrix(targets, predictions)
classification_rep = classification_report(targets, predictions)

print("Test accuracy: ", accuracy)
print("Confusion matrix: \n", confusion_mat)
print("Classification report: \n", classification_rep)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
ax1.plot(range(1, num_epoch + 1), train_losses, label='Training Loss', marker='o')
ax1.plot(range(1, num_epoch + 1), validation_losses, label='Validation Loss', marker='o')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss, with learning rate = {}'.format(lr))
ax1.legend()

ax2.plot(range(1, num_epoch + 1), train_acc_, label='Training Accuracy', marker='o')
ax2.plot(range(1, num_epoch + 1), val_acc_, label='Validation Accuracy', marker='o')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training and Validation Accuracy, with learning rate = {}'.format(lr))
ax2.legend()
plt.tight_layout()
plt.show()