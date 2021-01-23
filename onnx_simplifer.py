import onnx
from onnxsim import simplify

# load your predefined ONNX model
model = onnx.load('just_reshape.onnx')

input_shapes = {}

input_shape = ['input:2,3,4,5']

if input_shape is not None:
        for x in input_shape:
            if ':' not in x:
                input_shapes[None] = list(map(int, x.split(',')))
            else:
                pieces = x.split(':')
                # for the input name like input:0
                name, shape = ':'.join(
                    pieces[:-1]), list(map(int, pieces[-1].split(',')))
                input_shapes[name] = shape

# convert model
model_simp, check = simplify(model)

assert check, "Simplified ONNX model could not be validated"

onnx.save(model_simp, 'similifier.onnx')

# use model_simp as a standard ONNX model object