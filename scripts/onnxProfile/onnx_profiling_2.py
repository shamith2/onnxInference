# Script for profiling onnx model

import os
import sys

import onnx
import onnxruntime

from pathlib import Path

from onnxInsights.onnxHelpers import ONNXTransformer

workspace = Path(__file__).parent.resolve()

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

# for llm: llama
PHASE = 'DECODE'
BATCH_SIZE = 1
SEQ_LEN = 1024 if PHASE == 'PREFILL' else 1
MAX_LEN = 2048
CACHE_LEN = 1 if PHASE == 'PREFILL' else MAX_LEN - 1

onnx_t = ONNXTransformer(
    model_name='llama3_8b_fp16_' + PHASE.lower() + 'Phase',
    model_dir='llama3_8b_fp16'
)

uninferred_llm_onnx_model_path = os.path.join(workspace, 'models', 'rank_0_Meta-Llama-3-8B-Instruct_decoder_merged_model_fp16.onnx')

input_shapes = [(BATCH_SIZE, SEQ_LEN), (BATCH_SIZE, MAX_LEN), (BATCH_SIZE, SEQ_LEN)]

for i in range(32):
    input_shapes.append((BATCH_SIZE, 8, CACHE_LEN, 128)) # for key
    input_shapes.append((BATCH_SIZE, 8, CACHE_LEN, 128)) # for value

output_shapes = [(BATCH_SIZE, SEQ_LEN, 128256)]

for i in range(32):
    output_shapes.append((BATCH_SIZE, 8, MAX_LEN, 128)) # for key
    output_shapes.append((BATCH_SIZE, 8, MAX_LEN, 128)) # for value

inferred_onnx_model_path = onnx_t.shapeInfer(
    uninferred_llm_onnx_model_path,
    input_shapes,
    output_shapes
)

onnx_t.profileModel(inferred_onnx_model_path)

# onnx_t.profileModelonCPU(inferred_onnx_model_path)

# onnx_t.modifyGraph(delete_block=['DequantizeLinear', 'Clip', 'QuantizeLinear'], upper_2_ok=False, only_middle=True)
