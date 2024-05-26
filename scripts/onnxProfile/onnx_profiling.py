# Script for profiling onnx model

import os
import onnx

from pathlib import Path

from onnxInsights.onnxHelpers import ONNXTransformer

workspace = Path(__file__).parent.resolve()

onnx_t = ONNXTransformer()

uninferred_onnx_model_path = os.path.join(workspace, 'models', 'model.onnx')

inferred_onnx_model_path = onnx_t.shapeInfer('sd_unet', uninferred_onnx_model_path, [(1, 4, 64, 64), (1,), (1, 77, 1024)], [(1, 4, 64, 64)])

onnx_t.profileMemory(inferred_onnx_model_path)

# onnx_t.profileModel(inferred_onnx_model_path)

# onnx_t.modifyGraph(delete_block=['DequantizeLinear', 'Clip', 'QuantizeLinear'], upper_2_ok=False, only_middle=True)
