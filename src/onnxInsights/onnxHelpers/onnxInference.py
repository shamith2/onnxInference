# Implement Python functions to run ONNX Models inference using ONNXRuntime

import os
import collections
import time
from pathlib import Path

from typing import Optional, Union

import numpy
import onnxruntime

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__producer__ = "onnxruntimeInference"
__version__ = "0.1.0"


# ONNX Runtime Session Options for Inference
ORTSessionOptions = collections.namedtuple(
    'ORTSessionOptions',
    'execution_provider memory_device dynamic_axes intra_threads inter_threads graph_optimization_level execution_mode execution_order log_severity_level'
)

# ONNX Run Options for Inference
ORTNPUOptions = collections.namedtuple(
    'ORTRunOptions',
    'model_name instance_count layout config_file_path xclbin_path'
)

# Helper function
def getDtype(
        in_type: str
) -> numpy.ndarray:
    """
    Infer the correct dtype for inputs and outputs
    
    inputs:
        in_type: str = onnx model input dtype
    
    outputs:
        output: numpy.dtype = equivalent numpy dtype 
    """
    if in_type == "tensor(float16)":
        return numpy.float16

    elif in_type == "tensor(float)":
        return numpy.float32

    elif in_type == "tensor(long)" or in_type == "tensor(int64)":
        return numpy.int64
    
    elif in_type == "tensor(int32)":
        return numpy.int32

    elif in_type == "tensor(uint)":
        return numpy.uint8

    else:
        return numpy.float64


def init_NPU_Inference(
    npu_options: ORTNPUOptions,
) -> tuple[str, str, str]:
    """
    Initializing NPU enviroment variables
    """
    # NPU environment variables
    config_file_dir = os.path.join(os.environ['RYZEN_AI_INSTALLER'], 'voe-win_amd64-latest')
    xclbin_dir = os.path.join(os.environ['RYZEN_AI_INSTALLER'], 'voe-win_amd64-latest')

    working_dir = Path(__file__).parent.resolve()
    config_file_path = os.path.join(config_file_dir, 'vaip_config.json') if npu_options.config_file_path is None else npu_options.config_file_path
    cache_dir = os.path.join(working_dir.parent.parent, '.cache')
    cache_key = npu_options.model_name + '_ryzen_ai_' + config_file_path.split('\\')[-1][:-5] + '_' + npu_options.layout

    if npu_options.layout == '1x4':
        os.environ['XLNX_TARGET_NAME'] = "AMD_AIE2P_Nx4_Overlay"
        os.environ['XLNX_VART_FIRMWARE'] = os.path.join(xclbin_dir, 'AMD_AIE2P_Nx4_Overlay.xclbin') if npu_options.xclbin_path is None else npu_options.xclbin_path
        os.environ['NUM_OF_DPU_RUNNERS'] = str(min(npu_options.instance_count, 8))

    elif npu_options.layout == '4x4':
        os.environ['XLNX_TARGET_NAME'] = "AMD_AIE2P_4x4_Overlay"
        os.environ['XLNX_VART_FIRMWARE'] = os.path.join(xclbin_dir, 'AMD_AIE2P_4x4_Overlay.xclbin') if npu_options.xclbin_path is None else npu_options.xclbin_path
        os.environ['NUM_OF_DPU_RUNNERS'] = str(min(npu_options.instance_count, 2))

    else:
        raise Exception("Invalid Layout parameter: should be 1x4 or 4x4")

    os.environ['XLNX_ENABLE_CACHE'] = '1'            
    os.environ['XLNX_ONNX_EP_VERBOSE'] = '0'
    os.environ['XLNX_ENABLE_STAT_LOG'] = '0'

    if not os.path.exists(config_file_path):
        raise Exception("Cannot find {}".format(config_file_path))
    
    return cache_dir, config_file_path, cache_key


def processShape(
        shapeTensor: list,
        output_dynamic_shapes: Optional[Union[list[dict], dict]],
        idx: int
) -> list[int]:
    if isinstance(output_dynamic_shapes, dict):
        return [output_dynamic_shapes[value] if isinstance(value, str) else value for value in shapeTensor]

    elif isinstance(output_dynamic_shapes, list):
        return [output_dynamic_shapes[idx][value] if isinstance(value, str) else value for value in shapeTensor]
    
    else:
        return shapeTensor


def init_Inference(
        onnx_model_path: str,
        session_options: ORTSessionOptions,
        npu_options: Optional[ORTNPUOptions] = None
) -> tuple[list[numpy.ndarray], int]:
    """
    Initialize Inference
    """
    sess_options = onnxruntime.SessionOptions()

    sess_options.log_severity_level = session_options.log_severity_level
    sess_options.log_verbosity_level = 0

    sess_options.enable_cpu_mem_arena = True
    sess_options.enable_mem_pattern = True
    sess_options.enable_mem_reuse = True

    sess_options.intra_op_num_threads = session_options.intra_threads
    sess_options.inter_op_num_threads = session_options.inter_threads

    sess_options.enable_profiling = False
    sess_options.use_deterministic_compute = False

    # make dynamic axes static
    if session_options.dynamic_axes:
        for dim, value in session_options.dynamic_axes.items():
            sess_options.add_free_dimension_override_by_name(dim, value)

    # Graph Optimization Level
    sess_options.graph_optimization_level = session_options.graph_optimization_level

    # Execution Mode
    sess_options.execution_mode = session_options.execution_mode

    # ExecutionOrder
    sess_options.execution_order = session_options.execution_order

    # disable thread spinning
    sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")

    if session_options.execution_provider == 'NPU':
        # set environment variables
        cache_dir, config_file_path, cache_key = init_NPU_Inference(npu_options)

        ort_session = onnxruntime.InferenceSession(
            onnx_model_path,
            providers=['VitisAIExecutionProvider'],
            sess_options=sess_options,
            provider_options=[
                {
                    "config_file": config_file_path,
                    "cacheDir": cache_dir,
                    "cacheKey": cache_key,
                },
            ],
        )

        if 'VitisAIExecutionProvider' not in ort_session.get_providers():
            raise EnvironmentError(
                "ONNXRuntime does not support VitisAIExecutionProvider. Build ONNXRuntime appropriately")
    
    elif session_options.execution_provider == 'GPU':
        ort_session = onnxruntime.InferenceSession(
            onnx_model_path,
            providers=['DmlExecutionProvider'],
            sess_options=sess_options
        )

        if 'DmlExecutionProvider' not in ort_session.get_providers():
            raise EnvironmentError(
                "ONNXRuntime does not support DmlExecutionProvider. Build ONNXRuntime appropriately")

    else:
        ort_session = onnxruntime.InferenceSession(
            onnx_model_path,
            providers=['CPUExecutionProvider'],
            sess_options=sess_options
        )

    # Model compilation takes place here when the session is created

    # Disable CPU fallback
    ort_session.disable_fallback()

    return ort_session


def Inference(
        ort_session: onnxruntime.InferenceSession,
        model_inputs: tuple[Union[numpy.ndarray, onnxruntime.OrtValue]],
        memory_device: str,
        sustain_tensors: bool = False
) -> tuple[list[numpy.ndarray, float]]:
    """
    ONNX Model Inference using ONNX Runtime

    inputs:
        onnx_model_path: str = path to onnx model
        model_input: numpy.ndarray = input to the onnx model in sequential order. inputs are numpy arrays
    
    outputs:
        output: numpy.ndarray = output after inference in sequential order. outputs are numpy arrays
        inference_time: int = inference time, in nanoseconds, on onnxruntime session run()
    """
    # IO Binding
    io_binding = ort_session.io_binding()

    for i, inputTensor in enumerate(ort_session.get_inputs()):
        if memory_device == 'cpu' or not sustain_tensors:
            ortInput = onnxruntime.OrtValue.ortvalue_from_numpy(
                model_inputs[i],
                device_type=memory_device,
                device_id=0
            )
        
        else:
            ortInput = model_inputs[i]

        io_binding.bind_ortvalue_input(
            name=inputTensor.name,
            ortvalue=ortInput
        )

    for i, output in enumerate(ort_session.get_outputs()):
        io_binding.bind_output(
            name=output.name,
            device_type=memory_device,
            device_id=0,
            element_type=getDtype(output.type),
        )

    # benchmarking .run
    start = time.perf_counter_ns()

    ort_session.run_with_iobinding(io_binding)

    # inference time in ns
    inference_time = time.perf_counter_ns() - start

    if memory_device == 'cpu' or not sustain_tensors:
        outputs = io_binding.copy_outputs_to_cpu()

        io_binding.clear_binding_inputs()
        io_binding.clear_binding_outputs()
    
    else:
        outputs = io_binding.get_outputs()

    return outputs, inference_time
