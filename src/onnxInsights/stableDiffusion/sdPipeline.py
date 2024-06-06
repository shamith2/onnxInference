# Implement Python Stable Diffusion pipelines with ONNXRuntime and Python

import copy
import os
import time
import concurrent.futures
from pathlib import Path

from typing import Optional
from tqdm.auto import tqdm

import numpy
import torch
import onnxruntime

from transformers import CLIPTokenizerFast
from diffusers import EulerAncestralDiscreteScheduler

from ..onnxHelpers import ORTSessionOptions, init_Inference, Inference
from .sdHelper import changeDtype, dumpMetadata, saveTensorasImage

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__producer__ = "sdPipeline"
__version__ = "0.1.0"


# global variables: use with caution

ROOT = Path(__file__).parents[3].resolve()
WORKSPACE = Path(__file__).parent.resolve()

CACHE_DIR = os.path.join(ROOT, 'weights', 'stableDiffusion', '.cache')

SD_RESULT_DIR = os.path.join(ROOT, 'results', 'stableDiffusion', 'sd_results')
SD_T_RESULT_DIR = os.path.join(ROOT, 'results', 'stableDiffusion', 'sd_turbo_results')
SD_XL_RESULT_DIR = os.path.join(ROOT, 'results', 'stableDiffusion', 'sdxl_turbo_results') 

IMG_SIZE = 512
LATENT_SIZE = IMG_SIZE // 8
LATENT_CHANNELS = 4
VAE_DECODER_SCALE = 0.18215
VAE_ENCODER_SCALE = 0.18215
GUIDANCE_SCALE = 7.5


def generateLatents(
        shape: tuple[int],
        init_noise_sigma: float,
        seed: Optional[int] = None
) -> tuple[numpy.ndarray, float]:
    
    start = time.perf_counter_ns()

    rng = numpy.random.default_rng(seed)
    
    latents = rng.standard_normal(size=shape, dtype=numpy.float32) * init_noise_sigma

    end = time.perf_counter_ns()

    return latents, end


def runTextEncoder1(
        model_directory: str,
        inputTensor: numpy.ndarray,
        session_options: ORTSessionOptions,
        sustain_tensors: bool
) -> tuple[numpy.ndarray, float]:
    """
    Run Text Encoder onnx model

    Inputs:
        input: numpy.ndarray = int32[1, 77]

    Outputs:
        output: numpy.ndarray = float32[1, 77, 768]
        inference_time: int = inference time, in ms, of the model
    """
    seq_session_options = session_options._replace(dynamic_axes = {'sequence_length': 77})

    ort_session = init_Inference(
        os.path.join(model_directory, 'text_encoder', 'model.onnx'),
        session_options=seq_session_options
    )
    
    outputs, inference_time = Inference(
        ort_session,
        model_inputs=(changeDtype(inputTensor, "tensor(int32)"),),
        memory_device=seq_session_options.memory_device,
        sustain_tensors=sustain_tensors
    )

    inference_time = round(inference_time / 1e6, 3)

    return outputs, inference_time


def runTextEncoder2(
        model_directory: str,
        inputTensor: numpy.ndarray,
        session_options: ORTSessionOptions,
        sustain_tensors: bool
) -> tuple[numpy.ndarray, float]:
    """
    Run Text Encoder onnx model

    Inputs:
        input: numpy.ndarray = int64[1, 77]

    Outputs:
        output: numpy.ndarray = float32[1, 77, 1280]
        inference_time: int = inference time, in ms, of the model
    """
    seq_session_options = session_options._replace(dynamic_axes = {'sequence_length': 77})

    ort_session = init_Inference(
        os.path.join(model_directory, 'text_encoder_2', 'model.onnx'),
        session_options=seq_session_options
    )
    
    outputs, inference_time = Inference(
        ort_session,
        model_inputs=(changeDtype(inputTensor, "tensor(int64)"),),
        memory_device=seq_session_options.memory_device,
        sustain_tensors=sustain_tensors
    )

    inference_time = round(inference_time / 1e6, 3)

    return outputs, inference_time


def runVAEEncoder(
        model_directory: str,
        inputTensor: numpy.ndarray,
        session_options: ORTSessionOptions,
        sustain_tensors: bool
) -> tuple[numpy.ndarray, float]:
    """
    Run VAE Encoder onnx model

    Inputs:
        input: numpy.ndarray = float32[1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE]

    Outputs:
        latents: numpy.ndarray = float32[1, 3, IMG_SIZE, IMG_SIZE]
        inference_time: int = inference time, in ms, of the model
    """
    seq_session_options = session_options._replace(dynamic_axes = {'num_channels_latent': LATENT_CHANNELS, 'height_latent': LATENT_SIZE, 'width_latent': LATENT_SIZE, 'num_channels': 3})

    ort_session = init_Inference(
        os.path.join(model_directory, 'vae_encoder', 'model.onnx'),
        session_options=seq_session_options
    )

    outputs, inference_time = Inference(
        ort_session,
        model_inputs=(inputTensor,),
        memory_device=seq_session_options.memory_device,
        sustain_tensors=sustain_tensors
    )

    latents = outputs[0] * VAE_ENCODER_SCALE

    inference_time = round(inference_time / 1e6, 3)

    return latents, inference_time


def runVAEDecoder(
        model_directory: str,
        inputTensor: numpy.ndarray,
        session_options: ORTSessionOptions,
        sustain_tensors: bool
) -> tuple[numpy.ndarray, float]:
    """
    Run VAE Decoder onnx model

    Inputs:
        input: numpy.ndarray = float32[1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE]

    Outputs:
        output: numpy.ndarray = float32[1, 3, IMG_SIZE, IMG_SIZE]
        inference_time: int = inference time, in ms, of the model
    """
    seq_session_options = session_options._replace(dynamic_axes = {'num_channels_latent': LATENT_CHANNELS, 'height_latent': LATENT_SIZE, 'width_latent': LATENT_SIZE, 'num_channels': 3})

    ort_session = init_Inference(
        os.path.join(model_directory, 'vae_decoder', 'model.onnx'),
        session_options=seq_session_options
    )

    outputs, inference_time = Inference(
        ort_session,
        model_inputs=(inputTensor,),
        memory_device=seq_session_options.memory_device,
        sustain_tensors=sustain_tensors
    )

    inference_time = round(inference_time / 1e6, 3)

    return outputs[0], inference_time


def runUNet(
        model_directory: str,
        inputTensor1: numpy.ndarray,
        inputTensor2: numpy.ndarray,
        inputTensor3: numpy.ndarray,
        session_options: ORTSessionOptions,
        sustain_tensors: bool
) -> tuple[numpy.ndarray, float]:
    """
    Run UNet onnx model

    Inputs:
        input: numpy.ndarray = float32[1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE], int64[1], float32[1, 77, 1024]

    Outputs:
        output: numpy.ndarray = float32[1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE]
        inference_time: float = inference time, in ms, of the model
    """
    seq_session_options = session_options._replace(dynamic_axes = {'num_channels': LATENT_CHANNELS, 'height': LATENT_SIZE, 'width': LATENT_SIZE, 'sequence_length': 77})

    ort_session = init_Inference(
        os.path.join(model_directory, 'unet', 'model.onnx'),
        session_options=seq_session_options
    )

    outputs, inference_time = Inference(
        ort_session,
        model_inputs=(inputTensor1, inputTensor2, inputTensor3),
        memory_device=seq_session_options.memory_device,
        sustain_tensors=sustain_tensors
    )

    inference_time = round(inference_time / 1e6, 3)

    return outputs[0], inference_time


def runUNetXL(
        model_directory: str,
        inputTensor1: numpy.ndarray,
        inputTensor2: numpy.ndarray,
        inputTensor3: numpy.ndarray,
        inputTensor4: numpy.ndarray,
        inputTensor5: numpy.ndarray,
        session_options: ORTSessionOptions,
        sustain_tensors: bool
) -> tuple[numpy.ndarray, float]:
    """
    Run UNet onnx model

    Inputs:
        input: numpy.ndarray = float32[1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE], int64[1], float32[1, 77, 1024]

    Outputs:
        output: numpy.ndarray = float32[1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE]
        inference_time: float = inference time, in ms, of the model
    """
    seq_session_options = session_options._replace(dynamic_axes = {'num_channels': LATENT_CHANNELS, 'height': LATENT_SIZE, 'width': LATENT_SIZE, 'sequence_length': 77})

    ort_session = init_Inference(
        os.path.join(model_directory, 'unet', 'model.onnx'),
        session_options=seq_session_options
    )

    outputs, inference_time = Inference(
        ort_session,
        model_inputs=(inputTensor1, inputTensor2, inputTensor3, inputTensor4, inputTensor5),
        memory_device=seq_session_options.memory_device,
        sustain_tensors=sustain_tensors
    )

    inference_time = round(inference_time / 1e6, 3)

    return outputs[0], inference_time


def postProcess(
        model_directory: str,
        latents: numpy.ndarray,
        session_options: ORTSessionOptions,
        results_dir: str,
        filename: str = 'sd',
        sustain_tensors: bool = False,
        display: bool = False
):
    # taking into account the scale factor for VAE decoder
    latents /= VAE_DECODER_SCALE

    # run VAE decoder
    vae_output, vae_inference_time = runVAEDecoder(
        model_directory,
        latents,
        session_options,
        sustain_tensors
    )

    saveTensorasImage(vae_output, results_dir, filename, display=display)
    
    return vae_inference_time


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = numpy.std(noise_pred_text, axis=tuple(range(1, noise_pred_text.ndim)), keepdims=True)
    std_cfg = numpy.std(noise_cfg, axis=tuple(range(1, noise_cfg.ndim)), keepdims=True)

    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg

    return noise_cfg



def SD_pipeline(
        prompt: str,
        model_directory: str,
        steps: int = 10,
        cache_directory: Optional[str] = CACHE_DIR,
        save_directory: Optional[str] = SD_RESULT_DIR,
        output_filename: Optional[str] = 'sd_final_output'
) -> int:
    """
    Stable Diffusion v2 onnxruntime pipeline
    """
    seq_session_options = ORTSessionOptions('GPU', 'dml', {}, 2, 0, onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL,
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
    
    # initialization
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_directory, torch_dtype=torch.float16)

    scheduler = EulerAncestralDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    scheduler.set_timesteps(steps)

    if steps <= 1:
        scheduler.set_timesteps(2)
        scheduler.timesteps = scheduler.timesteps[0].unsqueeze(0)
    
    else:
       scheduler.set_timesteps(steps)

    inference_times = ()
    latent_norm_list = []

    # Tokenize and Encode Prompt
    cond_text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors='np')
    cond_input_ids = cond_text_input.input_ids

    uncond_text_input = tokenizer('', padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors='np')
    uncond_input_ids = uncond_text_input.input_ids

    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(generateLatents, (1, 4, LATENT_SIZE, LATENT_SIZE), scheduler.init_noise_sigma.item()),
                   executor.submit(runTextEncoder1, model_directory, cond_input_ids, seq_session_options, False),
                   executor.submit(runTextEncoder1, model_directory, uncond_input_ids, seq_session_options, False)]
        
        concurrent.futures.wait(futures)

        genLatent_ftr, cond_ftr, uncond_ftr = futures
        
        latents, _ = genLatent_ftr.result()
        cond_text_embedding, _ = cond_ftr.result()
        uncond_text_embedding, _ = uncond_ftr.result()

    text_embedding = numpy.concatenate((cond_text_embedding[0], uncond_text_embedding[0]), axis=0)
    
    # De-Noising Loop
    for i, t in enumerate(tqdm(scheduler.timesteps)):
        batch_latents = numpy.concatenate([latents] * 2, axis=0)
        batch_latents = scheduler.scale_model_input(torch.from_numpy(batch_latents), timestep=t).detach().numpy()

        unet_output, unet_inference_time = runUNet(
            model_directory,
            batch_latents,
            numpy.array([t], dtype=numpy.float32),
            text_embedding,
            seq_session_options,
            False
        )

        cond_noisy_output, uncond_noisy_output = numpy.split(unet_output, 2, axis=0)
        noisy_output = GUIDANCE_SCALE * cond_noisy_output + (1 - GUIDANCE_SCALE) * uncond_noisy_output

        if GUIDANCE_SCALE > 0:
            noisy_output = rescale_noise_cfg(noisy_output, cond_noisy_output, GUIDANCE_SCALE)

        latents = scheduler.step(torch.from_numpy(noisy_output), t, torch.from_numpy(latents)).prev_sample.detach().numpy()

        # just to view intermediate latents
        int_latents = copy.deepcopy(latents)

        latent_norm_list.append((t.item(), numpy.linalg.norm(numpy.reshape(int_latents, (int_latents.shape[0], -1)), axis=1).mean()))

        _ = postProcess(model_directory, int_latents, seq_session_options, os.path.join(save_directory, 'latents'), 'sd_int_' + str(t.item()), False, False)

        # average inference time for unet model for conditional input
        unet_inference_time += unet_inference_time

    unet_inference_time = unet_inference_time / len(scheduler.timesteps)

    vae_inference_time = postProcess(model_directory, latents, seq_session_options, os.path.join(save_directory, 'images'), output_filename, False, False)

    end = time.time() - start

    inference_times += (unet_inference_time, vae_inference_time, round(end * 1000, 3))

    dumpMetadata(
        inference_times,
        latent_norm_list,
        os.path.join(save_directory, 'logs'),
        "UNet,VAE Decoder,Total",
        filename='sd_pipeline'
    )

    return 0


def SD_Turbo_pipeline(
        prompt: str,
        model_directory: str,
        image: Optional[numpy.ndarray] = None,
        steps: int = 1,
        cache_directory: Optional[str] = CACHE_DIR,
        save_directory: Optional[str] = SD_T_RESULT_DIR,
        output_filename: Optional[str] = 'sd_final_output',
        display: bool = False
) -> int:
    """
    Stable Diffusion v2.1 Turbo onnxruntime pipeline
    """
    seq_session_options = ORTSessionOptions('GPU', 'dml', {}, 2, 0, onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL,
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
    
    # initialization
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_directory, torch_dtype=torch.float16)

    scheduler = EulerAncestralDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    if steps <= 1:
        scheduler.set_timesteps(2)
        scheduler.timesteps = scheduler.timesteps[:1]
    
    else:
       scheduler.set_timesteps(steps)

    latent_norm_list = []
    inference_times = ()

    # Tokenize and Encode Prompt
    cond_text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors='np')
    cond_input_ids = cond_text_input.input_ids

    cond_text_embedding, _ = runTextEncoder1(
        model_directory,
        cond_input_ids,
        seq_session_options,
        False
    )

    if image is not None:
        latents, _ = runVAEEncoder(
            model_directory,
            image,
            seq_session_options,
            False
        )

        # generate random noisy latents
        noise, _ = generateLatents(
            shape=(1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE),
            init_noise_sigma=scheduler.init_noise_sigma.item()
        )

        timestep = scheduler.timesteps if steps <= 1 else scheduler.timesteps[:1]

        latents = scheduler.add_noise(torch.from_numpy(latents), torch.from_numpy(noise), timestep).detach().numpy()
    
    else:
        # generate random noisy latents
        latents, _ = generateLatents(
            shape=(1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE),
            init_noise_sigma=scheduler.init_noise_sigma.item()
        )
    
    # De-Noising Loop
    print('\nImage de-noising loop...')
    for i, t in enumerate(tqdm(scheduler.timesteps)):
        batch_latents = scheduler.scale_model_input(torch.from_numpy(latents), timestep=t).detach().numpy()

        unet_output, unet_inference_time = runUNet(
            model_directory,
            batch_latents,
            numpy.array([t], dtype=numpy.int64),
            cond_text_embedding[0],
            seq_session_options,
            False
        )

        latents = scheduler.step(torch.from_numpy(unet_output), t, torch.from_numpy(latents)).prev_sample.detach().numpy()

        # just to view intermediate latents
        int_latents = copy.deepcopy(latents)

        latent_norm_list.append((t.item(), numpy.linalg.norm(numpy.reshape(int_latents, (int_latents.shape[0], -1)), axis=1).mean()))

        _ = postProcess(model_directory, int_latents, seq_session_options, os.path.join(save_directory, 'latents'), 'sd_int_' + str(t.item()), False, False)

        # average inference time for unet model for conditional input
        unet_inference_time += unet_inference_time

    unet_inference_time = unet_inference_time / steps

    vae_inference_time = postProcess(model_directory, latents, seq_session_options, os.path.join(save_directory, 'images'), output_filename, False, display)

    end = time.time() - start

    print('\n----- Total Inference Time for {} steps: {} -----\n'.format(steps, end))

    inference_times += (unet_inference_time, vae_inference_time, round(end * 1000, 3))

    dumpMetadata(
        inference_times,
        latent_norm_list,
        os.path.join(save_directory, 'logs'),
        "UNet,VAE Decoder,Total",
        filename='sd_pipeline'
    )

    return 0


def SDXL_Turbo_pipeline(
        prompt: str,
        model_directory: str,
        steps: int = 1,
        cache_directory: Optional[str] = CACHE_DIR,
        save_directory: Optional[str] = SD_XL_RESULT_DIR,
        output_filename: Optional[str] = 'sd_final_output',
        display: bool = False
) -> int:
    """
    Stable Diffusion XL Turbo onnxruntime pipeline
    """
    seq_session_options = ORTSessionOptions('GPU', 'dml', {}, 2, 0, onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL,
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
    
    # initialization
    tokenizer1 = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_directory, torch_dtype=torch.float16)
    tokenizer2 = CLIPTokenizerFast.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", cache_dir=cache_directory, torch_dtype=torch.float16)

    tokenizers = [tokenizer1, tokenizer2]
    text_encoders = [runTextEncoder1, runTextEncoder2]

    scheduler = EulerAncestralDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    if steps <= 1:
        scheduler.set_timesteps(2)
        scheduler.timesteps = scheduler.timesteps[:1]
    
    else:
       scheduler.set_timesteps(steps)

    latent_norm_list = []
    inference_times = ()

    cond_text_embeddings = []

    # Tokenize and Encode Prompt
    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        cond_text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors='np')
        cond_input_ids = cond_text_input.input_ids

        cond_text_embedding, _ = text_encoder(
            model_directory,
            cond_input_ids,
            seq_session_options,
            False
        )

        cond_text_embeddings.append(cond_text_embedding[-2])

        # only the pooler output from text encoder 2 is important
        pooled_cond_text_embedding = cond_text_embedding[0]

    cond_text_embeddings = numpy.concatenate(cond_text_embeddings, axis=-1)
    
    # generate random noisy latents
    latents, _ = generateLatents(
        shape=(1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE),
        init_noise_sigma=scheduler.init_noise_sigma.item()
    )
    
    time_ids = numpy.array(((IMG_SIZE, IMG_SIZE) + (0, 0) + (IMG_SIZE, IMG_SIZE),), dtype=numpy.float32)
    
    # De-Noising Loop
    print('\nImage de-noising loop...')
    for i, t in enumerate(tqdm(scheduler.timesteps)):
        batch_latents = scheduler.scale_model_input(torch.from_numpy(latents), timestep=t).detach().numpy()

        unet_output, unet_inference_time = runUNetXL(
            model_directory,
            batch_latents,
            numpy.array([t], dtype=numpy.int64),
            cond_text_embeddings,
            pooled_cond_text_embedding,
            time_ids,
            seq_session_options,
            False
        )

        latents = scheduler.step(torch.from_numpy(unet_output), t, torch.from_numpy(latents)).prev_sample.detach().numpy()

        # just to view intermediate latents
        int_latents = copy.deepcopy(latents)

        latent_norm_list.append((t.item(), numpy.linalg.norm(numpy.reshape(int_latents, (int_latents.shape[0], -1)), axis=1).mean()))

        _ = postProcess(model_directory, int_latents, seq_session_options, os.path.join(save_directory, 'latents'), 'sd_int_' + str(t.item()), False, False)

        # average inference time for unet model for conditional input
        unet_inference_time += unet_inference_time

    unet_inference_time = unet_inference_time / steps

    vae_inference_time = postProcess(model_directory, latents, seq_session_options, os.path.join(save_directory, 'images'), output_filename, False, display)

    end = time.time() - start

    print('\n----- Total Inference Time for {} steps: {} -----\n'.format(steps, end))

    inference_times += (unet_inference_time, vae_inference_time, round(end * 1000, 3))

    dumpMetadata(
        inference_times,
        latent_norm_list,
        os.path.join(save_directory, 'logs'),
        "UNet,VAE Decoder,Total",
        filename='sd_pipeline'
    )

    return 0

