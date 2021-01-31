#coding=utf-8
import onnx
import numpy as np
from onnx import helper
from onnx import numpy_helper

from test_tool import *

model = onnx.load("../model/just_reshape.onnx")
print('The model is:\n{}'.format(model))