#coding=utf-8
import onnx
import numpy as np
from onnx import helper
from onnx import numpy_helper

def get_input_names(model: onnx.ModelProto) -> List[str]:
    input_names = list(set([ipt.name for ipt in model.graph.input]) -
                       set([x.name for x in model.graph.initializer]))
    return input_names

class Tool(object):
    # 初始化onnx模型
    def __init__(self, onnx_model_path):
        self.model = onnx.load(onnx_model_path)
        self.model = onnx.shape_inference.infer_shapes(self.model)
        self.inputs = []
        self.outputs = []

    # 保存onnx模型
    def save(self, save_path):
        onnx.checker.check_model(self.model)
        self.model = onnx.shape_inference.infer_shapes(self.model)
        onnx.save(self.model, save_path)
    
    # 获取onnx模型的输入，返回一个列表
    def get_input_names(self):
        set_input = set()
        set_initializer = set()
        for ipt in self.model.graph.input:
            set_input.add(ipt.name)
        for x in model.graph.initializer:
            set_initializer.add(x.name)
        return list(set_input - set_initializer)
    
    # 为onnx模型增加batch维度
    def set_model_input_batch(self, index=0, name=None, batch_size=4):
        model_input = None
        if name is not None:
            for ipt in self.model.graph.input:
                if ipt.name == name:
                    model_input = ipt
        else:
            model_input = self.model.graph.input[index]
        if model_input:
            tensor_dim = model_input.type.tensor_type.shape.dim
            tensor_dim[0].ClearField("dim_param")
            tensor_dim[0].dim_value = batch_size
        else:
            print('get model input failed, check index or name')
        
    # 为onnx模型的输入设置形状
    def set_model_input_shape(self, index=0, name=None, shape=None):
        model_input = None
        if name is not None:
            for ipt in self.model.graph.input:
                if ipt.name == name:
                    model_input = ipt
        else:
            model_input = self.model.graph.input[index]
        if model_input:
            if shape is not None:
                tensor_shape_proto = model_input.type.tensor_type.shape
                tensor_shape_proto.ClearField("dim")
                tensor_shape_proto.dim.extend([])
                for d in shape:
                    dim = tensor_shape_proto.dim.add()
                    dim.dim_value = d
            else:
                print('get input shape failed, check input')
        else:
            print('get model input failed, check index or name')
    
    

    