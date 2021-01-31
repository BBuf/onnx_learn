# onnx_learn

记录学习ONNX的一些资料以及提供一些实践经验。

# 依赖

- numpy        必选
- onnx         必选
- onnxruntime  可选，如果脚本里面使用到了onnxruntime则需要安装
- pytorch      可选，如果脚本里面使用到了pytorch则需要安装

# 代码结构

- convert2onnx
    - pytorch2onnx_resize.py 通过Pytorch导出ONNX模型，Reshape操作

- onnxslim ONNX简化程序，来自大缺弦，[点这里](https://github.com/daquexian/onnx-simplifier)
- onnx_simplifer.py  调用onnxslim简化ONNX模型

- onnxapi
    - creat_onnx_example.py 创建一个onnx模型例子

- tools  维护一个工具类，方便修改ONNX模型来解决ONNX版本迭代以及框架之间对OP定义的不兼容问题。


    