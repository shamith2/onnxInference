# Script for profiling onnx model

import os
import sys

import onnx
import onnxruntime

from pathlib import Path

from onnxInsights.onnxHelpers import ONNXTransformer

workspace = Path(__file__).parent.resolve()

onnx_t = ONNXTransformer()

uninferred_onnx_model_path = os.path.join(workspace, 'models', 'model.onnx')

def get_shapes():
    dummy_session = onnxruntime.InferenceSession(
                    uninferred_onnx_model_path,
                    providers=["CPUExecutionProvider"]
                )
    
    print("Inputs: ")
    for _input in dummy_session.get_inputs():
        print(_input.name, _input.shape, _input.type)
    
    print("Outputs: ")
    for output in dummy_session.get_outputs():
        print(output.name, output.shape, output.type)
    
    sys.exit()

# get_shapes()

inferred_onnx_model_path = onnx_t.shapeInfer('sdxlt_unet', uninferred_onnx_model_path, [(1, 4, 64, 64), (1,), (1, 77, 2048), (1, 1280), (1, 6)], [(1, 4, 64, 64)])

onnx_t.profileMemory(inferred_onnx_model_path)

# onnx_t.profileModel(inferred_onnx_model_path)

# onnx_t.modifyGraph(delete_block=['DequantizeLinear', 'Clip', 'QuantizeLinear'], upper_2_ok=False, only_middle=True)
