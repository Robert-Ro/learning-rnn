# learning RNN

## train and test

```py
python train-test.py
```

## predict

```py
python main.py
```

## FAQ

### 图片识别错误

目前的模型是根据黑底白字的训练集来做的训练，所以待识别的图片必须是黑底白字，不然识别不了

## TODO

- [ ] 理解各行代码的意义
- [ ] 如何使用迁移学习
- [ ] pytorch 使用 gpu
- [ ] 闭环： 手写->摄像头输入->图片预处理->模型识别
