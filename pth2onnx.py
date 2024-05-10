import torch
import torch.onnx
import torchvision.models as models
from models import FastDVDnet
# 设置使用CPU导出模型
device = torch.device("cpu")

def convert():
    # 模型定义来自于torchvision，样例生成的模型文件是基于resnet50模型
    model = FastDVDnet()
    MFDN = torch.load('model_24.722_SNR_10.pth', map_location='cpu')    #根据实际文件名称修改
    model.load_state_dict(MFDN)

    # batch_size = 21  #批处理大小
    batch_size = 1  #批处理大小
    input_shape = (5, 64, 64)   #输入数据,改成自己的输入shape

    # 模型设置为推理模式
    model.eval()

    dummy_input = torch.randn(batch_size, *input_shape) #  定义输入shape
    torch.onnx.export(model,
                      dummy_input,
                      "fastdvdnet_snr_10.onnx",
                      input_names = ["input"],   # 构造输入名
                      output_names = ["output"],    # 构造输出名
                      opset_version=11)    # ATC工具目前支持opset_version=9，10，11，12，13
                      # dynamic_axes={"input":{0:"batch_size"}, "output":{0:"batch_size"}})  #支持输出动态轴

if __name__ == "__main__":
    convert()