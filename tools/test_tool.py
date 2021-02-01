#coding=utf-8
import onnx
import numpy as np
from onnx import helper
from onnx import numpy_helper

from test_tool import *

model = onnx.load("../model/tiny-yolov3-11.onnx")
onnx.checker.check_model(model)
print('The model is:\n{}'.format(model))