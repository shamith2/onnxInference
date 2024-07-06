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

def get_shapes(model_path):
    dummy_session = onnxruntime.InferenceSession(
                    model_path,
                    providers=["CPUExecutionProvider"]
                )
    
    print("Inputs: ")
    for _input in dummy_session.get_inputs():
        print(_input.name, _input.shape, _input.type)
    
    print("Outputs: ")
    for output in dummy_session.get_outputs():
        print(output.name, output.shape, output.type)
    
    sys.exit()


# get_shapes(uninferred_onnx_model_path)

# onnx_t = ONNXTransformer(model_name='sdxl_turbo_unet')

# inferred_onnx_model_path = onnx_t.shapeInfer('sdxlt_unet', uninferred_onnx_model_path, [(1, 4, 64, 64), (1,), (1, 77, 2048), (1, 1280), (1, 6)], [(1, 4, 64, 64)])

# for llm: tiny llama
PHASE = 'TOKEN'
BATCH_SIZE = 1
SEQ_LEN = 1024 if PHASE == 'PROMPT' else 1
MAX_LEN = 2048
CACHE_LEN = 1 if PHASE == 'PROMPT' else MAX_LEN - 1

onnx_t = ONNXTransformer(model_name='tinyllama')

uninferred_llm_onnx_model_path = os.path.join(workspace, 'models', 'model_quantized.onnx')

input_shapes = [(BATCH_SIZE, SEQ_LEN), (BATCH_SIZE, MAX_LEN), (BATCH_SIZE, SEQ_LEN)]

for i in range(22):
    input_shapes.append((BATCH_SIZE, 4, CACHE_LEN, 64)) # for key
    input_shapes.append((BATCH_SIZE, 4, CACHE_LEN, 64)) # for value

output_shapes = [(BATCH_SIZE, SEQ_LEN, 32000)]

for i in range(22):
    output_shapes.append((BATCH_SIZE, 4, MAX_LEN, 64)) # for key
    output_shapes.append((BATCH_SIZE, 4, MAX_LEN, 64)) # for value

inferred_onnx_model_path = onnx_t.shapeInfer(
    'tinyllama',
    uninferred_llm_onnx_model_path,
    input_shapes,
    output_shapes
)

onnx_t.profileModel(inferred_onnx_model_path)

# onnx_t.profileModelonCPU(inferred_onnx_model_path)

# onnx_t.modifyGraph(delete_block=['DequantizeLinear', 'Clip', 'QuantizeLinear'], upper_2_ok=False, only_middle=True)
