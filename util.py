import torch
import torch.nn as nn
from torchvision import transforms
import os
import cv2

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
# # TODO use gpu is has gpu
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


def pre_handle(image):
    """_summary_
    图片背景/内容颜色反转
    Args:
        image (string): 待处理的图片路径

    Returns:
        string: 处理好的图片路径
    """
    [name, extension] = os.path.basename(image).split(".")
    file_path = os.path.dirname(image)
    image = cv2.imread(image)
    inverted_image = cv2.bitwise_not(image)
    new_image = f"{file_path}/{name}_inverted.{extension}"
    cv2.imwrite(new_image, inverted_image)
    return new_image
