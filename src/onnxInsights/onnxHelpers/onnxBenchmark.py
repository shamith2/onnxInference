# This script contains functions for converting PyTorch model to ONNX, benchmarking ONNX models using ONNXRuntime, Quantizing and Performance Analysis of ONNX nodels
# Version: v1.0.0

from collections import deque
from datetime import datetime
import gc
import json
import math
import numpy as np
import os
import sys
import shutil
import re
from pathlib import Path
import logging
import time
import timeit
from concurrent.futures import wait, ThreadPoolExecutor
import functools

import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantFormat, QuantType, CalibrationDataReader
from onnxruntime.quantization.shape_inference import quant_pre_process

import torch
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__producer__ = "onnxBenchmark"
__version__ = "1.0.0"


# global functions
def check_install_dir(voe_dir: str):
    if not os.path.exists(os.path.join(os.environ['RYZEN_AI_INSTALLER'], voe_dir)):
        raise Exception("Invalid VOE installation: voe-4.0-win_amd64 or voe-win_amd64-latest missing")
        sys.exit(9)
    
    return True

# infer the correct dtype for inputs and outputs
def get_correct_dtype(input_type: str):
    if input_type == "(float16)":
        return np.float16

    elif input_type == "(float)":
        return np.float32

    elif input_type == "(double)":
        return np.double

    elif input_type == "(long)":
        return np.int_

    else:
        return


# generate random input images instead of real images; batch size = 1
# input data size := (b, h, w, c)
def get_random_input(
        input_shape: list[int],
        input_type: str,
        batch_size: int = 1
):
    input_data = 255.0 * np.random.random_sample([batch_size] + input_shape[1:]).astype(
        get_correct_dtype(re.search(r"\((.*)\)", input_type).group(0)))

    return input_data


# check if tensor is nhcw
def is_nchw(shape):
    if shape[1] < shape[2] and shape[1] < shape[3]:
        return True

    else:
        return False


# CalibrationData for quantization
class ImageCalibrationData(CalibrationDataReader):
    def __init__(
            self,
            onnx_model_path: str,
            num_input_data: int = 5,
            batch_size: int = 1,
            sample_size: int = 1,
            calib_dataset: bool = True
    ):
        self.calib_dataset = calib_dataset

        dummy_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

        self.input_dict = {}
        for _input in dummy_session.get_inputs():
            if self.calib_dataset and is_nchw(_input.shape):
                self.input_dict[str(_input.name)] = (self.convert_nchw_to_nhwc_shape(_input.shape), _input.type)
            else:
                self.input_dict[str(_input.name)] = (_input.shape, _input.type)

        if batch_size != 1:
            logger.warning("\nbatch size has to be 1. Setting batch size to 1 ...\n")
            self.bz = 1
        else:
            self.bz = batch_size

        self.num_total_inputs = num_input_data
        self.samples = sample_size

        # initial input data batch
        self.input_feed = self.get_input_batch()

    def convert_nchw_to_nhwc_shape(self, shape):
        shape[1], shape[3] = shape[3], shape[1]

        return shape

    def data_init(self):
        # input data size := (b, h, w, c)
        for (input_shape, input_type) in self.input_dict.values():
            yield np.zeros([0] + input_shape[1:],
                           dtype=get_correct_dtype(re.search(r"\((.*)\)", input_type).group(0)))

    def get_input_batch(self):
        input_batch = []

        for _ in range(self.samples):
            input_feed = {}

            for (input_name, input_data), input_i in zip(self.input_dict.items(), self.data_init()):
                input_feed[input_name] = np.concatenate((input_i,
                                                         get_random_input(*input_data, self.bz)), axis=0)

            input_batch.append(input_feed)

        self.num_total_inputs -= self.samples

        if self.calib_dataset:
            return iter(input_batch)
        else:
            return input_batch

    def get_next(self):
        next_input = next(self.input_feed, None)

        if next_input:
            return next_input

        del self.input_feed
        gc.collect()

        if self.num_total_inputs >= self.samples:
            self.input_feed = self.get_input_batch()

            return next(self.input_feed, None)

        else:
            return None


class ImageData(ImageCalibrationData):
    def __init__(
            self,
            onnx_model_path: str,
            num_input_data: int = 100,
            batch_size: int = 1,
            sample_size: int = 100,
            calib_dataset: bool = False,
    ):
        super(ImageData, self).__init__(onnx_model_path, num_input_data, batch_size, sample_size, calib_dataset)

    def get_input_data(self):
        if self.input_feed is None:
            return self.get_input_batch()

        return self.input_feed


def generate_random_data(
        onnx_model_path,
        num_input_data: int = 100,
        batch_size: int = 1,
        sample_size: int = 100,
        calib_dataset: bool = False,
):
    data_generator = ImageData(onnx_model_path, num_input_data=num_input_data, batch_size=batch_size,
                               sample_size=sample_size, calib_dataset=calib_dataset)

    return data_generator.get_input_data()


class ONNXInference:
    def __init__(
            self,
            model_name: str = None,
            metadata: str = 'test',
            model_path: Optional[str] = None,
            mode: Optional[str] = None
    ):
        if model_name is None:
            raise Exception("model_name cannot be None")
            sys.exit(1)

        self.model_name = str(model_name)

        self.working_dir = Path(__file__).parent.resolve()

        self.workspace = os.path.join(self.working_dir, 'onnx')
        self.cache_dir = os.path.join(self.workspace, '.cache')

        self.config_file_dir = None
        self.xclbin_dir = None
        self.instance_count = None
        self.runtime = None
        self.iterations = None
        self.opset_version = None
        self.use_external_data = None

        self.fp32_onnx_dir = os.path.join(self.workspace, 'onnx_models', 'fp32', self.model_name)
        self.fp32_onnx_dir_p = os.path.join(self.fp32_onnx_dir, 'partial')
        self.fp32_infer_onnx_dir = os.path.join(self.workspace, 'onnx_models', 'fp32_infer', self.model_name)
        self.quant_onnx_dir = os.path.join(self.workspace, 'onnx_models', 'quant', self.model_name)
        self.ryzen_ai_onnx_dir = os.path.join(self.workspace, 'onnx_models', 'ryzen_ai', self.model_name)
        self.fp32_results_dir = os.path.join(self.workspace, 'results', 'fp32', self.model_name, metadata)
        self.quant_results_dir = os.path.join(self.workspace, 'results', 'quant', self.model_name, metadata)
        self.ryzen_ai_results_dir = os.path.join(self.workspace, 'results', 'ryzen_ai', self.model_name, metadata)
        self.fp32_prof_dir = os.path.join(self.workspace, 'profiling', 'fp32', self.model_name)
        self.quant_prof_dir = os.path.join(self.workspace, 'profiling', 'quant', self.model_name)
        self.ryzen_ai_prof_dir = os.path.join(self.workspace, 'profiling', 'ryzen_ai', self.model_name)

        for i in [self.workspace, self.cache_dir, self.fp32_onnx_dir, self.fp32_onnx_dir_p, self.fp32_infer_onnx_dir,
                  self.quant_onnx_dir, self.ryzen_ai_onnx_dir, self.fp32_results_dir, self.quant_results_dir,
                  self.ryzen_ai_results_dir, self.fp32_prof_dir, self.quant_prof_dir, self.ryzen_ai_prof_dir]:
            os.makedirs(i, exist_ok=True)

        # copy onnx model or cache to appropriate directory if model_dir is provided
        if model_path is not None and os.path.exists(model_path):
            model_path = os.path.realpath(model_path)

            if mode not in ['fp32', 'int8', 'ryzen-ai', 'cache']:
                raise Exception("mode has to be either fp32 or int8 or ryzen-ai or cache")
                sys.exit(2)

            if mode == 'fp32':
                to_path = os.path.join(self.fp32_onnx_dir, self.model_name + '.onnx')
                logging.info("Copying {} to {}\n".format(model_path, to_path))
                shutil.copy2(model_path, to_path)

            elif mode == 'int8':
                to_path = os.path.join(self.quant_onnx_dir, self.model_name + '_int8.onnx')
                logging.info("Copying {} to {}\n".format(model_path, to_path))
                shutil.copy2(model_path, to_path)

            elif mode == 'ryzen-ai':
                to_path = os.path.join(self.ryzen_ai_onnx_dir, self.model_name + '_int8.onnx')
                logging.info("Copying {} to {}\n".format(model_path, to_path))
                shutil.copy2(model_path, to_path)

            # if mode is cache, then model_path is a directory
            else:
                if os.path.isdir(model_path):
                    to_path = os.path.join(self.cache_dir, self.model_name + metadata)
                    logging.info("Copying {} to {}\n".format(model_path, to_path))
                    shutil.copytree(model_path, to_path, copy_function=shutil.copy2, dirs_exist_ok=True)
                
                else:
                    raise Exception("If mode is cache, then model_path has to be a directory")
                    sys.exit(3)

    def convert_torch_to_onnx(self,
                              model: torch.nn.Module,
                              pass_inputs: bool = False,
                              model_inputs: tuple[torch.Tensor] = None,
                              input_shape: tuple = None,
                              input_names: Optional[list] = None,
                              output_names: Optional[list] = None,
                              input_dynamic_axes: Optional[list[dict]] = None,
                              output_dynamic_axes: Optional[list[dict]] = None,
                              opset_version: int = 17,
                              use_dynamo: bool = False,
                              use_external_data: bool = False,
                              exist_ok: bool = True,
    ) -> int:
        if not use_dynamo:
            self.opset_version = opset_version
            self.use_external_data = use_external_data

            if self.opset_version > 17:
                logger.warning("Opset version cannot be greater than 17 if not using torch dynamo. Setting opset "
                               "version to 17 ...\n")
                self.opset_version = 17

        else:
            logger.warning("Dynamic Axes might not work as intended\n")
            self.opset_version = 18
            self.use_external_data = False

        if not isinstance(model, torch.nn.Module):
            raise Exception("[ERROR] Model has to be of type torch.nn.Module")
            sys.exit(4)

        if os.path.exists(self.fp32_onnx_dir):
            if not exist_ok:
                shutil.rmtree(self.fp32_onnx_dir)
                os.mkdir(self.fp32_onnx_dir)

            else:
                logger.info("ONNX directory already exists. Skipping this step.")
                return 1

        # Export the model to ONNX
        if self.use_external_data:
            onnx_model_path_p = os.path.join(self.fp32_onnx_dir_p, self.model_name + '_partial.onnx')

        onnx_model_path = os.path.join(self.fp32_onnx_dir, self.model_name + '.onnx')

        if pass_inputs and model_inputs is None:
            raise Exception("Input cannot be None")
            sys.exit(5)

        if not pass_inputs:
            if input_shape is None:
                raise Exception("Input shape cannot be None")
                sys.exit(6)

            else:
                model_inputs = tuple(torch.randn(*input_shape, requires_grad=False))

        with torch.no_grad():
            # set model to eval
            model.eval()

            logger.info("Converting {} from PyTorch to ONNX ...\n".format(self.model_name.capitalize()))
            # convert pytorch model to onnx
            if not use_dynamo:
                dynamic_axes = {}

                for i in range(len(input_dynamic_axes)):
                    dynamic_axes[str(input_names[i])] = input_dynamic_axes[i]

                for i in range(len(output_dynamic_axes)):
                    dynamic_axes[str(output_names[i])] = output_dynamic_axes[i]

                torch.onnx.export(
                    model,
                    model_inputs,
                    onnx_model_path_p if self.use_external_data else onnx_model_path,
                    export_params=True,
                    do_constant_folding=True,
                    opset_version=self.opset_version,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    verbose=True,
                    training=torch.onnx.TrainingMode.EVAL,
                    operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                )

            else:
                torch._dynamo.config.dynamic_shapes = True
                torch._dynamo.config.capture_scalar_outputs = True
                torch._dynamo.config.automatic_dynamic_shapes = True

                kwargs = {}

                for i in range(len(input_names)):
                    kwargs[str(input_names[i])] = model_inputs[i]

                export_options = torch.onnx.ExportOptions(dynamic_shapes=True)

                export_output = torch.onnx.dynamo_export(model, **kwargs, export_options=export_options)
                export_output.save(onnx_model_path)

        if self.use_external_data:
            logger.info("Saving external data to one file ...\n")

            # try freeing memory
            gc.collect()

            onnx_model = onnx.load(onnx_model_path_p, load_external_data=True)

            onnx.save_model(
                onnx_model,
                onnx_model_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=self.model_name + "-data",
                size_threshold=1024,
            )

        try:
            onnx.checker.check_model(onnx_model_path)

        except onnx.checker.ValidationError as e:
            raise Exception(e)
            sys.exit(7)

        logger.info("Successfully converted PyTorch model to ONNX!!\n")

        return 0

    def quantize(
            self,
            shape_infer: bool = True,
            external_data_format: bool = False
    ) -> int:
        import vai_q_onnx

        input_model_path = os.path.join(self.fp32_onnx_dir, self.model_name + '.onnx')

        # configure model paths
        if shape_infer:
            infer_model_path = os.path.join(self.fp32_infer_onnx_dir, self.model_name + '_infer.onnx')

        else:
            infer_model_path = input_model_path

        external_data_location = os.path.dirname(infer_model_path) if external_data_format else None

        quantized_model_path = os.path.join(self.ryzen_ai_onnx_dir, self.model_name + '_int8.onnx')

        # shape inference
        if shape_infer:
            logger.info("Performing Shape Inference ...\n")

            quant_pre_process(
                input_model_path,
                infer_model_path,
                skip_optimization=external_data_format,
                skip_onnx_shape=False,
                skip_symbolic_shape=False,
                int_max=(2 ** 31 - 1),
                verbose=1,
                save_as_external_data=external_data_format,
                all_tensors_to_one_file=external_data_format,
                external_data_location=external_data_location,
                external_data_size_threshold=1024,
            )
        
        dummy_session = ort.InferenceSession(infer_model_path, providers=['CPUExecutionProvider'])

        is_model_nchw = False
        
        for _input in dummy_session.get_inputs():
            if is_nchw(_input.shape):
                is_model_nchw = True
            
            else:
                is_model_nchw = False

        logger.info("Quantizing onnx model ...\n")

        vai_q_onnx.quantize_static(
            infer_model_path,
            quantized_model_path,
            calibration_data_reader=ImageCalibrationData(infer_model_path, 5, 1, 1, True),
            quant_format=QuantFormat.QDQ,
            calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            enable_ipu_cnn=True,  # option for model running on ipu
            extra_options={'ActivationSymmetric': True},
            convert_nchw_to_nhwc=is_model_nchw,
        )

        try:
            onnx.checker.check_model(quantized_model_path)

        except onnx.checker.ValidationError as e:
            raise Exception(e)
            sys.exit(8)

        logger.info("Successfully quantized onnx model!!\n")

        return 0

    def start_processing_timed(
            self,
            latency_per_iteration: deque,
            ort_session: ort.InferenceSession,
            output_feed: list,
            input_feed: dict,
            runtime: int,
            profiling: bool
    ) -> None:
        # ignore 1st run
        if not profiling:
            _, i_time = timeit.timeit(lambda: ort_session.run(output_feed, input_feed), number=2)
            latency_per_iteration.append(i_time)

        # run benchmark for %runtime% seconds
        latency_per_iteration.append(timeit.repeat(lambda: ort_session.run(output_feed, input_feed), number=1,
                                                   repeat=math.ceil(runtime / i_time)))

    def start_processing_iter(
            self,
            ort_session: ort.InferenceSession,
            output_feed: list,
            input_feed: dict,
    ) -> float:
        iter_time = timeit.timeit(lambda: ort_session.run(output_feed, input_feed), number=1)

        return iter_time

    def start_inference(self,
                        instance_count: int = 1,
                        layout: Optional[str] = '1x4',
                        config_file_name: Optional[str] = None,
                        compile_only: bool = False,
                        num_input_data: Optional[int] = 100,
                        benchmark: bool = False,
                        profiling: bool = False,
                        disable_thread_spinning: bool = True,
                        runtime: int = 60,
                        iterations: Optional[int] = None,
                        num_threads: int = 8,
                        inf_mode: str = 'fp32',
                        verbosity: int = 3,
                        intra_threads: int = 1
    ):
        if inf_mode == 'ryzen-ai':
            onnx_model_path = os.path.join(self.ryzen_ai_onnx_dir, self.model_name + '_int8.onnx')
            results_dir = self.ryzen_ai_results_dir
            prof_dir = self.ryzen_ai_prof_dir

        elif inf_mode == 'int8':
            onnx_model_path = os.path.join(self.quant_onnx_dir, self.model_name + '_int8.onnx')
            results_dir = self.quant_results_dir
            prof_dir = self.quant_prof_dir

        else:
            onnx_model_path = os.path.join(self.fp32_onnx_dir, self.model_name + '.onnx')
            results_dir = self.fp32_results_dir
            prof_dir = self.fp32_prof_dir

        self.instance_count = instance_count
        self.runtime = runtime

        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = verbosity

        sess_options.intra_op_num_threads = intra_threads
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        logging.info("Disabling thread spinning ...\n")
        if disable_thread_spinning:
            sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        else:
            sess_options.add_session_config_entry("session.intra_op.allow_spinning", "1")

        if benchmark:
            sess_options.log_severity_level = 3

        if profiling:
            sess_options.enable_profiling = True
            sess_options.log_severity_level = verbosity

        logger.info("ONNX Model Path: {}\n".format(onnx_model_path))

        # ryzen-ai := int8 on ryzen ai processor
        if inf_mode == 'ryzen-ai':
            voe_dir = 'voe-win_amd64-latest'
            check_install_dir(voe_dir)

            self.config_file_dir = os.path.join(os.environ['RYZEN_AI_INSTALLER'], voe_dir)
            self.xclbin_dir = os.path.join(os.environ['RYZEN_AI_INSTALLER'], voe_dir)

            if layout == '1x4':
                os.environ['XLNX_TARGET_NAME'] = "AMD_AIE2P_Nx4_Overlay"
                os.environ['XLNX_VART_FIRMWARE'] = os.path.join(self.xclbin_dir, 'AMD_AIE2P_Nx4_Overlay.xclbin')
                os.environ['NUM_OF_DPU_RUNNERS'] = str(min(self.instance_count, 8))

            elif layout == '4x4':
                os.environ['XLNX_TARGET_NAME'] = "AMD_AIE2P_4x4_Overlay"
                os.environ['XLNX_VART_FIRMWARE'] = os.path.join(self.xclbin_dir, 'AMD_AIE2P_4x4_Overlay.xclbin')
                os.environ['NUM_OF_DPU_RUNNERS'] = str(min(self.instance_count, 2))

            else:
                raise Exception("Invalid Layout parameter: should be 1x4 or 4x4")
                sys.exit(10)

            os.environ['XLNX_ENABLE_CACHE'] = '1'            
            os.environ['XLNX_ONNX_EP_VERBOSE'] = '1' if compile_only else '0'
            os.environ['XLNX_ENABLE_STAT_LOG'] = '0'

            if config_file_name is None:
                config_file_name = 'vaip_config_1x4.json'
            else:
                config_file_name = config_file_name + '_' + layout + '.json'

            config_file_path = os.path.join(self.config_file_dir, str(config_file_name))

            if not os.path.exists(config_file_path):
                raise Exception("Cannot find {} in {}".format(config_file_name, config_file_path))
                sys.exit(11)

            ort_session = ort.InferenceSession(
                onnx_model_path,
                providers=["VitisAIExecutionProvider"],
                sess_options=sess_options,
                provider_options=[
                    {
                        "config_file": config_file_path,
                        "cacheDir": self.cache_dir,
                        "cacheKey": self.model_name + '_ryzen_ai_' + str(config_file_name)[:-5],
                    },
                ],
            )

            if "VitisAIExecutionProvider" not in ort_session.get_providers():
                raise EnvironmentError(
                    "ONNXRuntime does not support VitisAIExecutionProvider. Build ONNXRuntime appropriately")
                sys.exit(12)

            # IPU compilation takes place when the session is created
            logger.info("Model compiled successfully!!\n")

        else:
            config_file_path = None
            cache_dir = None

            ort_session = ort.InferenceSession(
                onnx_model_path,
                providers=["CPUExecutionProvider"],
                sess_options=sess_options,
            )

        if compile_only:
            return 0

        # Disable CPU fallback
        ort_session.disable_fallback()

        # model outputs
        output_feed = [x.name for x in ort_session.get_outputs()]

        task_queue = deque()
        results = deque()

        # local function
        def task_fn(queue, sampler, num_input_samples, bz, sample_size):
            queue.append(sampler(onnx_model_path, num_input_samples, bz, sample_size, calib_dataset=False)[0])

        logger.info("Creating task queue...\n")

        # create a task_queue
        with ThreadPoolExecutor(max_workers=max(num_threads, 8)) as threads:
            tasks = [threads.submit(task_fn, task_queue, generate_random_data, 1, 1, 1) for _ in range(num_input_data)]
            # wait(tasks)

        task = functools.partial(self.start_processing_iter, ort_session, output_feed)

        logger.info("Starting ONNXRuntime Benchmark...\n")
        logger.info("Minimize Terminal for Power Measurement: 15 seconds\n")
        time.sleep(15)

        # run benchmark inference
        with ThreadPoolExecutor(max_workers=num_threads) as threads:
            start = time.perf_counter()

            while time.perf_counter() - start <= self.runtime:
                for result in threads.map(task, task_queue):
                    results.append(result)

            # threads.shutdown(wait=True, cancel_futures=True)

        if profiling:
            prof_file = ort_session.end_profiling()
            logger.info("Successfully completed ONNXRuntime Profiling!!\n")

        logger.info("Successfully completed ONNXRuntime Benchmark!!\n")

        throughput, result_data = self.get_latency_result(list(results))

        _datetime = datetime.now().strftime("%m%d%Y%H%M%S")

        metadata = '{}_onnx_{}n_{}sec_{}i_{}nt_{}it_{}dts_{}l_{}_{}fps_{}'.format(self.model_name, str(num_input_data),
                                                                                  str(self.runtime),
                                                                                  str(self.instance_count),
                                                                                  str(num_threads), str(intra_threads),
                                                                                  str(int(disable_thread_spinning)),
                                                                                  str(layout),
                                                                                  str(config_file_name)[:-5],
                                                                                  str(throughput), _datetime)

        result_log = os.path.join(results_dir, metadata + '.json')

        with open(result_log, 'w', encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4)

        if profiling:
            shutil.move(os.path.join(os.getcwd(), prof_file),
                        os.path.join(prof_dir, prof_file[:-5] + metadata + '.json'))
            logger.info(
                "Saved profiling log in {}\n".format(os.path.join(prof_dir, prof_file[:-5] + metadata + '.json')))

        logger.info("Saved results log in {}\n".format(result_log))

        gc.collect()

        return 0

    # adapted from https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/benchmark_helper.py
    def get_latency_result(self, latency_list: list, batch_size: int = 1):
        latency_list_ms = np.array(latency_list, dtype=np.float64) * 1000

        inferences = len(latency_list_ms)
        latency_ms = np.sum(latency_list_ms) / inferences
        latency_variance = np.var(latency_list_ms, dtype=float)
        throughput = inferences * batch_size / float(self.runtime)

        uniq, inv = np.unique(np.round(latency_list_ms, 2), return_inverse=True)
        mode_latency_ms = uniq[np.bincount(inv).argmax()]

        return round(throughput), {
            "inferences": inferences,
            "latency_variance": f"{latency_variance:.2f}",
            "min_latency": f"{min(latency_list_ms):.2f}",
            "mode_latency": f"{mode_latency_ms:.2f}",
            "latency_90_percentile": f"{np.percentile(latency_list_ms, 90):.2f}",
            "latency_95_percentile": f"{np.percentile(latency_list_ms, 95):.2f}",
            "latency_99_percentile": f"{np.percentile(latency_list_ms, 99):.2f}",
            "average_latency_ms": f"{latency_ms:.2f}",
            "IPS": f"{throughput:.2f}",
        }
