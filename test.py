from torchvision import transforms
from PIL import Image
import torch
from util import RNN

learning_rate = 0.001
sequence_length = 28
hidden_size = 128
num_classes = 10
batch_size = 64
input_size = 28
num_layers = 2
num_epochs = 3


# 加载已经训练好的模型
model = RNN(input_size, hidden_size, num_layers, num_classes)
model.load_state_dict(torch.load("rnn_model.pth"))
model.eval()

# 图像预处理
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),  # 将图像转换为灰度
        transforms.Resize((28, 28)),  # 调整图像大小为模型输入大小
        transforms.ToTensor(),  # 转换为张量
    ]
)

# 读取待识别的图像
image_path = "./temp/testing/1/2.png"
image = Image.open(image_path)
image = transform(image)

# 添加批次维度（模型通常接受批次作为输入）
image = image.unsqueeze(0)

# 使用模型进行推断
with torch.no_grad():
    output = model(image)

# 获取预测结果
_, predicted = torch.max(output, 1)
predicted_digit = predicted.item()

print(f"Predicted Digit: {predicted_digit}")
