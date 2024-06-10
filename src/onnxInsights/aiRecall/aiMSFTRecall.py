# Implement Python Custom AI Recall pipelines with ONNXRuntime and Python

import copy
import os
import time
import concurrent.futures
from pathlib import Path

from typing import Optional
from tqdm.auto import tqdm

import math
import numpy
import torch
import onnxruntime

from transformers import CLIPTokenizerFast
from diffusers import EulerAncestralDiscreteScheduler

from ..onnxHelpers import ORTSessionOptions, ORTNPUOptions, init_Inference, Inference
from .sdHelper import changeDtype, siLU, dumpMetadata, getTensorfromImage, saveTensorasImage

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__producer__ = "aiRecallPipeline"
__version__ = "0.1.0"


# global variables: use with caution
ROOT = Path(__file__).parents[3].resolve()
WORKSPACE = Path(__file__).parent.resolve()

CACHE_DIR = os.path.join(ROOT, 'weights', 'aiRecall', '.cache')

IMG_SIZE = 512


def analyseScreenshots(
        directory: str
):
    """
    Analyse and understand what an user is doing in the screenshot
    """
    for file in os.listdir(directory):
        screenshot = os.path.join(directory, file)
    

def runImageEncoder(
    model_directory: str,
    inputTensor1: numpy.ndarray,
    inputTensor2: numpy.ndarray,
    session_options: ORTSessionOptions,
    sustain_tensors: bool
) -> tuple[numpy.ndarray, float]:
    """
    Run Image Encoder onnx model

    Inputs:
        pixel_values: numpy.ndarray = float32[num_images, max_num_crops, 3, height, width]
        image_sizes: numpy.ndarray: int64[num_images, 2]

    Outputs:
        output: numpy.ndarray = float32[1, 3, IMG_SIZE, IMG_SIZE]
        inference_time: int = inference time, in ms, of the model
    """
    seq_session_options = session_options._replace(dynamic_axes = {'num_images': 1, 'max_num_crops': 3, 'height': IMG_SIZE, 'width': IMG_SIZE})

    ort_session = init_Inference(
        os.path.join(model_directory, 'phi-3-v-128k-instruct-vision.onnx'),
        session_options=seq_session_options
    )

    outputs, inference_time = Inference(
        ort_session,
        model_inputs=(inputTensor1, changeDtype(inputTensor2, "tensor(int64)")),
        memory_device=session_options.memory_device,
        sustain_tensors=sustain_tensors
    )

    inference_time = round(inference_time / 1e6, 3)

    return outputs[0], inference_time


def runTextEncoder(
    model_directory: str,
    inputTensor: numpy.ndarray,
    session_options: ORTSessionOptions,
    sustain_tensors: bool
) -> tuple[numpy.ndarray, float]:
    """
    Run Text Encoder onnx model

    Inputs:
        input_ids: numpy.ndarray = int64[batch_size, sequence_length]

    Outputs:
        output: numpy.ndarray = float32[batch_size, sequence_length, 3072]
        inference_time: int = inference time, in ms, of the model
    """
    seq_session_options = session_options._replace(dynamic_axes = {'batch_size': 1, 'sequence_length': 77})

    ort_session = init_Inference(
        os.path.join(model_directory, 'phi-3-v-128k-instruct-text-embedding.onnx'),
        session_options=seq_session_options
    )

    outputs, inference_time = Inference(
        ort_session,
        model_inputs=(changeDtype(inputTensor, "tensor(int64)"),),
        memory_device=session_options.memory_device,
        sustain_tensors=sustain_tensors
    )

    inference_time = round(inference_time / 1e6, 3)

    return outputs[0], inference_time


def AI_Recall_pipeline(
) -> int:
    """
    Custom AI Recall onnxruntime pipeline
    """
    seq_session_options = ORTSessionOptions('CPU', 'cpu', {}, 2, 0, onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL,
                                            onnxruntime.ExecutionMode.ORT_SEQUENTIAL,
                                            onnxruntime.ExecutionOrder.DEFAULT, 3)
    
    acc_seq_session_options = ORTSessionOptions('GPU', 'dml', {}, 2, 0, onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL,
                                                onnxruntime.ExecutionMode.ORT_SEQUENTIAL,
                                                onnxruntime.ExecutionOrder.DEFAULT, 3)
    
    prl_session_options = ORTSessionOptions('CPU', 'cpu', {}, 2, 1, onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL,
                                            onnxruntime.ExecutionMode.ORT_PARALLEL,
                                            onnxruntime.ExecutionOrder.DEFAULT, 3)
    
    for dir in ['logs', 'images', 'latents']:
        path = os.path.join(save_directory, dir)
        
        if not os.path.exists(path):
            os.makedirs(path)

    start = time.time()
    
    

    end = time.time() - start

    print('\n----- Total Inference Time for {} steps: {} s -----\n'.format(steps, end))

    # inference times in ms
    inference_times += (round(end * 1000, 3))

    return 0

