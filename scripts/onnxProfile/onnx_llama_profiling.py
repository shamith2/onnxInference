# Script for profiling onnx model

import os
import sys

import onnxruntime

from pathlib import Path

from onnxInsights.onnxHelpers import ONNXProfiler
from onnxInsights.onnxHelpers import memoryView

root = Path(__file__).parents[2].resolve()
workspace = Path(__file__).parent.resolve()


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


# for llm: llama3
PHASE = 'DECODEN'
BATCH_SIZE = 1
SEQ_LEN = 1024 if PHASE == 'PREFILL' else 1
MAX_LEN = 2048
CACHE_LEN = 1 if PHASE == 'PREFILL' else MAX_LEN - 1

onnx_t = ONNXProfiler(
    model_name='llama3_8b_fp16_' + PHASE.lower() + 'Phase',
    model_dir='llama3_8b_fp16'
)


def shape_infer():
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
        None,
        input_shapes,
        output_shapes
    )

    return inferred_onnx_model_path


# inferred_onnx_model_path = shape_infer()

inferred_onnx_model_path = os.path.join(root, 'results', 'onnxProfile', 'models', 'llama3_8b_fp16',
                                        'llama3_8b_fp16_decodenPhase_inferred.onnx')

# onnx_t.profileModel(inferred_onnx_model_path)

local_memory_view = memoryView(
    model_dir='llama3_8b_fp16',
    model_profile='llama3_8b_fp16_decodenPhase_summary.csv',
    outputs_profile='llama3_8b_fp16_decodenPhase_track_output_summary.csv'
)

local_memory_view.run_with_cache(
    memory_size=3,
    cache_size=0,
    final_outputs=('logits'),
    plot_memory=True
)

# onnx_t.profileModelonCPU(inferred_onnx_model_path)

# onnx_t.modifyGraph(delete_block=['DequantizeLinear', 'Clip', 'QuantizeLinear'], upper_2_ok=False, only_middle=True)

