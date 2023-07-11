# AI_lab5
The code of lab5 of AI course.
## Setup
This implemetation is based on Python3.10. To run the code, you need the following dependencies:
keras~=2.10.0
numpy~=1.25.0
pandas~=1.5.3
matplotlib~=3.7.1
scikit-learn~=1.2.2
You can simply run
`pip install -r requirements.txt`
## Repository structure
`|-- dataset # 提供的数据集，包括训练集和测试集
|-- result_screenshot # 一些控制台输出实验结果的截图
|-- ResNet.py # 模型中ResNet结构的代码
|-- cnn_resnet.png
|-- cnn_resnet.py # 实验的主体代码，包括数据预处理、模型的训练和测试以及消融实验等
|-- cnn_resnet_params.png
|-- pred.txt # 测试集的预测结果
`
## Run code
`python cnn_resnet.py`
