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
    
    # 通过名字获取onnx模型中的计算节点
    def get_node_by_name(self, name):
        for node in self.model.graph.node:
            if node.name == name:
                return node
    
    # 通过op的类型获取onnx模型的计算节点
    def get_nodes_by_optype(self, typename):
        nodes = []
        for node in self.model.graph.node:
            if node.op_type == typename:
                nodes.append(node)
        return nodes

    # 通过名字获取onnx模型计算节点的权重
    def get_weight_by_name(self, name):
        for weight in self.model.graph.initializer:
            if weight.name == name:
                return weight
    
    # 设置权重，注意这个weight是TensorProto类型，`https://github.com/onnx/onnx/blob/b1e0bc9a31eaefc2a9946182fbad939843534984/onnx/onnx.proto#L461`
    def set_weight(self, weight, data_numpy=None, all_ones=False, all_zeros=False):
        if data_numpy is not None:
            raw_shape = tuple([i for i in weight.dims])
            new_shape = np.shape(data_numpy)
            if weight.data_type == 8:
                print("Can NOT handle string data type right now...")
                exit()
            if new_shape != raw_shape:
                print("Warning: the new weight shape is not consistent with original shape!")
                weight.dims[:] = list(new_shape)
                for model_input in self.model.graph.input:
                    if model_input.name == weight.name:
                        # copy from onnx.helper...
                        tensor_shape_proto = model_input.type.tensor_type.shape
                        tensor_shape_proto.ClearField("dim")
                        tensor_shape_proto.dim.extend([])
                        for d in new_shape:
                            dim = tensor_shape_proto.dim.add()
                            dim.dim_value = d

            weight.ClearField("float_data")
            weight.ClearField("int32_data")
            weight.ClearField("int64_data")
            weight.raw_data = data_numpy.tobytes()
        else:
            if all_ones:
                wr = numpy_helper.to_array(weight)
                wn = np.ones_like(wr)
            elif all_zeros:
                wr = numpy_helper.to_array(weight)
                wn = np.zeros_like(wr)
            else:
                print("You must give a data_numpy to set the weight, or set the all_ones/all_zeros flag.")
                exit()
            weight.ClearField("float_data")
            weight.ClearField("int32_data")
            weight.ClearField("int64_data")
            weight.raw_data = wn.tobytes()

    # 通过名字设置ONNX节点的权重
    def set_weight_by_name(self, name, data_numpy=None, all_ones=False, all_zeros=False):
        weight = self.get_weight_by_name(name)
        self.set_weight(weight, data_numpy, all_ones, all_zeros)
    
    # 移除ONNX模型中的目标节点
    def remove_node(self, target_node):
        '''
            删除只有一个输入和输出的节点
        '''
        node_input = target_node.input[0]
        node_output = target_node.output[0]
        # 将后继节点的输入设置为目标节点的前置节点
        for node in self.model.graph.node:
            for i, n in enumerate(node.input):
                if n == node_output:
                    node.input[i] = node_input

        target_names = set(target_node.input) & set([weight.name for weight in self.model.graph.initializer])
        self.remove_weights(target_names)
        target_names.add(node_output)
        self.remove_inputs(target_names)
        self.remove_value_infos(target_names)
        self.model.graph.node.remove(target_node)

    # 移除ONNX模型中指定节点的权重
    def remove_weights(self, name_list):
        rm_list = []
        for weight in self.model.graph.initializer:
            if weight.name in name_list:
                rm_list.append(weight)
        for weight in rm_list:
            self.model.graph.initializer.remove(weight)

    # 移除ONNX模型中指定的输入节点
    def remove_inputs(self, name_list):
        rm_list = []
        for input_t in self.model.graph.input:
            if input_t.name in name_list:
                rm_list.append(input_t)
        for input_t in rm_list:
            self.model.graph.input.remove(input_t)

    # 移除ONNX模型中指定的输入输出节点
    def remove_value_infos(self, name_list):
        rm_list = []
        for value_info in self.model.graph.value_info:
            if value_info.name in name_list:
                rm_list.append(value_info)
        for value_info in rm_list:
            self.model.graph.value_info.remove(value_info)

    