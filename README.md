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
- onnxapi
    - creat_onnx_example.py 创建一个onnx模型例子
- tools  维护一个工具类，方便修改ONNX模型来解决ONNX版本迭代以及框架之间对OP定义的不兼容问题
- test_tool.py 测试ONNX工具类

# 学习笔记

- [ONNX初探](https://mp.weixin.qq.com/s/H1tDcmrg0vTcSw9PgpgIIQ)
- [ONNX再探](https://mp.weixin.qq.com/s/_iNhfZNR5-swXLhHKjYRkQ)
- [onnx simplifier 和 optimizer](https://mp.weixin.qq.com/s/q0Aa2LRpeCPCnIzRJbMmaQ)
- [onnx2pytorch和onnx-simplifier新版介绍](https://mp.weixin.qq.com/s/NDv-quXeBrPeDcCbg97FHA)
- [深度学习框架OneFlow是如何和ONNX交互的？](https://mp.weixin.qq.com/s/sxBDHl00jAKRXq-Y6Rii7A)
- [Pytorch转ONNX-理论篇](https://mp.weixin.qq.com/s/RoqaMPwCbtHfLKgnJX95ng)
- [Pytorch转ONNX-实战篇1（tracing机制）](https://mp.weixin.qq.com/s/L2lZAo35ZeybuiH3tgJsvw)
- [Pytorch转ONNX-实战篇2（实战踩坑总结）](https://mp.weixin.qq.com/s/nG45SDO2_J48omSkn27EtQ)