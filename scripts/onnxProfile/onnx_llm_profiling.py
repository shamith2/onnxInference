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


# for llms
PHASE = 'DECODEN'
BATCH_SIZE = 1
SEQ_LEN = 1024 if PHASE == 'PREFILL' else 1
MAX_LEN = 2048
CACHE_LEN = 1 if PHASE == 'PREFILL' else MAX_LEN - 1
KV_CHANNELS = 1 # 8
KV_EMBED_SIZE = 256 # 128
OUTPUT_EMBED_SIZE = 256000 # 128256

# onnx_t = ONNXProfiler(
#     model_name='llama3_8b_fp16_' + PHASE.lower() + 'Phase',
#     model_dir='llama3_8b_fp16'
# )

onnx_t = ONNXProfiler(
    model_name='gemma1.1_2b_fp16_' + PHASE.lower() + 'Phase',
    model_dir='gemma1.1_2b_fp16'
)


def shape_infer():
    uninferred_llm_onnx_model_path = os.path.join(workspace, 'models', 'rank_0_Meta-Llama-3-8B-Instruct_decoder_merged_model_fp16.onnx')
    # uninferred_llm_onnx_model_path = os.path.join(workspace, 'models', 'rank_0_gemma-1.1-2b-it_decoder_merged_model_fp16.onnx')

    input_shapes = [(BATCH_SIZE, SEQ_LEN), (BATCH_SIZE, MAX_LEN), (BATCH_SIZE, SEQ_LEN)]

    for i in range(32):
        input_shapes.append((BATCH_SIZE, KV_CHANNELS, CACHE_LEN, KV_EMBED_SIZE)) # for key
        input_shapes.append((BATCH_SIZE, KV_CHANNELS, CACHE_LEN, KV_EMBED_SIZE)) # for value

    output_shapes = [(BATCH_SIZE, SEQ_LEN, OUTPUT_EMBED_SIZE)]

    for i in range(32):
        output_shapes.append((BATCH_SIZE, KV_CHANNELS, MAX_LEN, KV_EMBED_SIZE)) # for key
        output_shapes.append((BATCH_SIZE, KV_CHANNELS, MAX_LEN, KV_EMBED_SIZE)) # for value

    inferred_onnx_model_path = onnx_t.shapeInfer(
        uninferred_llm_onnx_model_path,
        None,
        input_shapes,
        output_shapes
    )

    return inferred_onnx_model_path


# inferred_onnx_model_path = shape_infer()

# inferred_onnx_model_path = os.path.join(root, 'results', 'onnxProfile', 'models', 'llama3_8b_fp16',
#                                         'llama3_8b_fp16_decodenPhase_inferred.onnx')

inferred_onnx_model_path = os.path.join(root, 'results', 'onnxProfile', 'models', 'gemma1.1_2b_fp16',
                                        'gemma1.1_2b_fp16_decodenPhase_inferred.onnx')

# onnx_t.profileModel(inferred_onnx_model_path)

# local_memory_view = memoryView(
#     model_dir='llama3_8b_fp16',
#     model_profile='llama3_8b_fp16_decodenPhase_summary.csv',
#     outputs_profile='llama3_8b_fp16_decodenPhase_track_output_summary.csv'
# )

local_memory_view = memoryView(
    model_dir='gemma1.1_2b_fp16',
    model_profile='gemma1.1_2b_fp16_decodenPhase_summary.csv',
    outputs_profile='gemma1.1_2b_fp16_decodenPhase_track_output_summary.csv'
)

for local_memory_size in [3, 40, 80]: # range(1, 20 + 1, 1):
    score = local_memory_view.run_with_cache(
        local_memory_size=local_memory_size,
        cache_size=0,
        final_outputs=('logits'),
        plot_memory=True
    )

    print("Local Memory Size: {}, Score: {}\n".format(local_memory_size, score))


# onnx_t.profileModelonCPU(inferred_onnx_model_path)

# onnx_t.modifyGraph(delete_block=['DequantizeLinear', 'Clip', 'QuantizeLinear'], upper_2_ok=False, only_middle=True)

