import torch

from torchvision import datasets, transforms
from util import RNN, device

# Hyper parameters
learning_rate = 0.001
sequence_length = 28
hidden_size = 128
num_classes = 10
batch_size = 64
input_size = 28
num_layers = 2
num_epochs = 3

train_dataset = datasets.MNIST(
    root="./data/training", train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = datasets.MNIST(
    root="./data/testing", train=False, transform=transforms.ToTensor(), download=True
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)


model = RNN(input_size, hidden_size, num_layers, num_classes)
to_device(model, device)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                    epoch + 1, num_epochs, i + 1, total_step, loss.item()
                )
            )


# Evaluate the model
model.eval()
with torch.no_grad():
    right = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        right += (predicted == labels).sum().item()
print(
    "Test Accuracy of the model on the 10000 test images: {} %".format(
        100 * right / total
    )
)

torch.save(model.state_dict(), "rnn_model.pth")
