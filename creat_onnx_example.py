import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


# # The protobuf definition can be found here:
# # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto


# # Create one input (ValueInfoProto)
# X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3, 2])
# pads = helper.make_tensor_value_info('pads', TensorProto.FLOAT, [1, 4])

# value = helper.make_tensor_value_info('value', AttributeProto.FLOAT, [1])


# # Create one output (ValueInfoProto)
# Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3, 4])

# # Create a node (NodeProto) - This is based on Pad-11
# node_def = helper.make_node(
#     'Pad', # node name
#     ['X', 'pads', 'value'], # inputs
#     ['Y'], # outputs
#     mode='constant', # attributes
# )

# # Create the graph (GraphProto)
# graph_def = helper.make_graph(
#     [node_def],
#     'test-model',
#     [X, pads, value],
#     [Y],
# )

# # Create the model (ModelProto)
# model_def = helper.make_model(graph_def, producer_name='onnx-example')

# print('The model is:\n{}'.format(model_def))
# onnx.checker.check_model(model_def)
# print('The model is checked!')

model = onnx.load('tiny-yolov3-11.onnx')

print(model)