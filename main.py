from PIL import Image
from util import (
    transform,
    RNN,
    sequence_length,
    input_size,
    hidden_size,
    num_layers,
    num_classes,
    device,
)
import torch

# 加载已训练的模型
model = RNN(input_size, hidden_size, num_layers, num_classes)
# model.load_state_dict(torch.load("rnn_model.pth", map_location=device))
model.load_state_dict(torch.load("rnn_model_demo1.pth", map_location=device))
model.eval()
# model.to(device)


# 读取待识别的图像
image_path = "./temp/training/7/38.png"
image_path = "./test/4/4.png"  # 白底黑字，识别不了
image_path = "./test/3/1.png"

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
