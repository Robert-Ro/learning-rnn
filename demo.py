import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

# 超参数
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 5
learning_rate = 0.001

# 数据预处理，将图像调整为模型的输入尺寸，并转换为张量
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((sequence_length, input_size)),
        transforms.ToTensor(),
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义RNN模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# 加载本地数据集
# train_dataset = ImageFolder(root="./temp/training", transform=transform)
# train_loader = torch.utils.data.DataLoader(
#     dataset=train_dataset, batch_size=batch_size, shuffle=True
# )
# model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # 训练模型
# total_step = len(train_loader)
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         images = images.reshape(-1, sequence_length, input_size).to(device)
#         labels = labels.to(device)

#         # 前向传播
#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         # 反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if (i + 1) % 100 == 0:
#             print(
#                 f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}"
#             )

# print("训练完成")

# # 保存模型
# torch.save(model.state_dict(), "rnn_model.pth")


# 加载本地测试数据集
# test_dataset = ImageFolder(root="./temp/testing", transform=transform)
# test_loader = torch.utils.data.DataLoader(
#     dataset=test_dataset, batch_size=batch_size, shuffle=False
# )

# # 加载已训练的模型
model = RNN(input_size, hidden_size, num_layers, num_classes)
model.load_state_dict(torch.load("rnn_model.pth"))
model.eval()  # 设置模型为评估模式


model.to(device)

# 测试模型
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.reshape(-1, sequence_length, input_size).to(device)
#         labels = labels.to(device)

#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f"Accuracy on test images: {100 * correct / total:.2f}%")


# 读取待识别的图像
image_path = "./temp/testing/9/9.png"
image = Image.open(image_path)
image = transform(image)

# 添加批次维度（模型通常接受批次作为输入）
image = image.unsqueeze(0)

image = image.to(device)

# 使用模型进行推断
with torch.no_grad():
    output = model(image.view(1, sequence_length, input_size))

# 获取预测结果
_, predicted = torch.max(output, 1)
predicted_digit = predicted.item()

print(f"Predicted Digit: {predicted_digit}")
