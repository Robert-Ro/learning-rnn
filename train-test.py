from util import (
    RNN,
    transform,
    batch_size,
    learning_rate,
    num_epochs,
    sequence_length,
    input_size,
    hidden_size,
    num_classes,
    num_layers,
    num_epochs,
)
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn

# # TODO use gpu is has gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
开始训练
加载数据集，"dataset" 文件夹包含了多个子文件夹，每个子文件夹代表一个类别
"""
train_dataset = ImageFolder(root="./temp/training", transform=transform)

# 创建数据加载器

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
model = RNN(input_size, hidden_size, num_layers, num_classes)
model = model.to(device)
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
print("训练完成")


# 开始评估
test_dataset = ImageFolder(root="./temp/testing", transform=transform)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True
)

# Evaluate the model
# # 加载已训练的模型
# model = RNN(input_size, hidden_size, num_layers, num_classes)
# model.load_state_dict(torch.load("rnn_model.pth"))
# model.to(device)
model.eval()  # 设置模型为评估模式

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
