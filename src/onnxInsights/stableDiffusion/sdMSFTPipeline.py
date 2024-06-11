# Implement Python Stable Diffusion pipelines with ONNXRuntime and Python
# Adapted from onnxInsights sdPipeline.py

import copy
import os
import time
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
from ..onnxHelpers import changeDtype, siLU, dumpMetadata, getTensorfromImage, saveTensorasImage

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__producer__ = "sdMSFTPipeline"
__version__ = "0.1.0"


# global variables: use with caution
ROOT = Path(__file__).parents[3].resolve()
WORKSPACE = Path(__file__).parent.resolve()

CACHE_DIR = os.path.join(ROOT, 'weights', 'stableDiffusion', '.cache')

SD_T_RESULT_DIR = os.path.join(ROOT, 'results', 'stableDiffusion', 'sd_msft_results')

IMG_SIZE = 512
LATENT_SIZE = IMG_SIZE // 8
LATENT_CHANNELS = 4
TIME_EMBEDDING_SIZE = 320
VAE_DECODER_SCALE = 0.18215
VAE_ENCODER_SCALE = 0.18215


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


def getTimestepEmbedding(
    timestep: float,
    linearLayer1_w: numpy.ndarray,
    linearLayer1_b: numpy.ndarray,
    linearLayer2_w: numpy.ndarray,
    linearLayer2_b: numpy.ndarray,
    embedding_dim: int = TIME_EMBEDDING_SIZE,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000
) -> numpy.ndarray:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings

    Inputs:
        timestep: float = timestep of scheduler/sampler step
        embedding_dim: int = the dimension of the output
        max_period: int = minimum frequency
    
    Output:
        t_emb: numpy.ndarray[float] = sinusoidal timestep positional embedding
    """
    timesteps = numpy.array([timestep], dtype=numpy.float32)

    half_dim = embedding_dim // 2
    full_dim = embedding_dim * 4

    exponent_scale = -math.log(max_period) / (half_dim - downscale_freq_shift)

    exponent = exponent_scale * numpy.arange(
        start=0, stop=half_dim, dtype=numpy.float32
    )

    t_emb = numpy.exp(exponent)
    t_emb = timesteps[:, None] * t_emb[None, :]

    # scale embeddings
    t_emb = scale * t_emb

    # concat sine and cosine embeddings
    t_emb = numpy.concatenate((numpy.sin(t_emb), numpy.cos(t_emb)), axis=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        t_emb = numpy.concatenate((t_emb[:, half_dim:], t_emb[:, :half_dim]), axis=-1)

    # create random weights and bias for linear layers
    # rng = numpy.random.default_rng()

    # scale_1 = math.sqrt(1.0 / embedding_dim)
    # linearLayer1_w = rng.uniform(low=-scale_1, high=scale_1, size=(full_dim, embedding_dim))
    # linearLayer1_b = rng.uniform(low=-scale_1, high=scale_1, size=(full_dim,))

    # scale_2 = math.sqrt(1.0 / full_dim)
    # linearLayer2_w = rng.uniform(low=-scale_2, high=scale_2, size=(full_dim, full_dim))
    # linearLayer2_b = rng.uniform(low=-scale_2, high=scale_2, size=(full_dim,))

    # compute timestep embedding: linearLayer1 + siLU + linearLayer2
    t_emb = siLU(t_emb @ linearLayer1_w.T + linearLayer1_b) @ linearLayer2_w.T + linearLayer2_b

    return t_emb.astype(numpy.float32)


def runTextEncoder(
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
        output: numpy.ndarray = float32[1, 77, 1024]
        inference_time: int = inference time, in ms, of the model
    """
    psq1_npu_options = ORTNPUOptions('psq1', 1, '4x4', None, None)
    psq2_npu_options = ORTNPUOptions('psq2', 1, '4x4', None, None)
    
    psq1_ort_session = init_Inference(
        os.path.join(model_directory, 'text_encoder', 'model.onnx'),
        session_options=session_options,
        npu_options=psq1_npu_options if session_options.execution_provider == 'NPU' else None
    )

    psq2_ort_session = init_Inference(
        os.path.join(model_directory, 'text_encoder_2', 'model.onnx'),
        session_options=session_options,
        npu_options=psq2_npu_options if session_options.execution_provider == 'NPU' else None
    )
    
    psq1_outputs, psq1_inference_time = Inference(
        psq1_ort_session,
        model_inputs=(changeDtype(inputTensor, "tensor(int64)"),),
        memory_device=session_options.memory_device,
        sustain_tensors=sustain_tensors
    )

    outputs, psq2_inference_time = Inference(
        psq2_ort_session,
        model_inputs=psq1_outputs,
        memory_device=session_options.memory_device,
        sustain_tensors=sustain_tensors
    )

    return outputs[0], round(psq1_inference_time / 1e6, 3), round(psq2_inference_time / 1e6, 3)


def runVAEEncoder(
        model_directory: str,
        inputTensor: numpy.ndarray,
        session_options: ORTSessionOptions,
        sustain_tensors: bool
) -> tuple[numpy.ndarray, float]:
    """
    Run PST VAE Encoder onnx model

    Inputs:
        input: numpy.ndarray = float32[1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE]

    Outputs:
        latents: numpy.ndarray = float32[1, 3, IMG_SIZE, IMG_SIZE]
        inference_time: int = inference time, in ms, of the model
    """
    vae_encoder_npu_options = ORTNPUOptions('vae_encoder', 1, '4x4', None, None)

    ort_session = init_Inference(
        os.path.join(model_directory, 'vae_encoder', 'model.onnx'),
        session_options=session_options,
        npu_options=vae_encoder_npu_options if session_options.execution_provider == 'NPU' else None
    )

    outputs, inference_time = Inference(
        ort_session,
        model_inputs=(inputTensor,),
        memory_device=session_options.memory_device,
        sustain_tensors=sustain_tensors
    )

    inference_time = round(inference_time / 1e6, 3)

    latents = outputs[0] * VAE_ENCODER_SCALE

    return latents, inference_time


def runVAEDecoder(
        model_directory: str,
        inputTensor: numpy.ndarray,
        session_options: ORTSessionOptions,
        sustain_tensors: bool
) -> tuple[numpy.ndarray, float]:
    """
    Run PSS VAE Decoder onnx model

    Inputs:
        input: numpy.ndarray = float32[1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE]

    Outputs:
        output: numpy.ndarray = float32[1, 3, IMG_SIZE, IMG_SIZE]
        inference_time: int = inference time, in ms, of the model
    """
    vae_decoder_npu_options = ORTNPUOptions('vae_decoder', 1, '4x4', None, None)

    ort_session = init_Inference(
        os.path.join(model_directory, 'vae_decoder', 'model.onnx'),
        session_options=session_options,
        npu_options=vae_decoder_npu_options if session_options.execution_provider == 'NPU' else None
    )

    outputs, inference_time = Inference(
        ort_session,
        model_inputs=(inputTensor,),
        memory_device=session_options.memory_device,
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
        input: numpy.ndarray = float32[1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE], float32[1, 1280], float32[1, 77, 1024]

    Outputs:
        output: numpy.ndarray = float32[1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE]
        inference_time: float = inference time, in ms, of the model
    """
    unet_npu_options = ORTNPUOptions('unet', 1, '4x4', None, None)

    ort_session = init_Inference(
        os.path.join(model_directory, 'unet', 'model.onnx'),
        session_options=session_options,
        npu_options=unet_npu_options if session_options.execution_provider == 'NPU' else None
    )

    outputs, inference_time = Inference(
        ort_session,
        model_inputs=(inputTensor1, inputTensor2, inputTensor3),
        memory_device=session_options.memory_device,
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
    vae_output, vae_decoder_inference_time = runVAEDecoder(
        model_directory,
        latents,
        session_options,
        sustain_tensors
    )

    saveTensorasImage(vae_output, results_dir, filename, display=display, normalize=False)
    
    return vae_decoder_inference_time


def SD_Turbo_MSFT_pipeline(
        prompt: str,
        model_directory: str,
        weights_directory: str,
        image: Optional[numpy.ndarray] = None,
        steps: int = 4,
        cache_directory: Optional[str] = CACHE_DIR,
        save_directory: Optional[str] = SD_T_RESULT_DIR,
        output_filename: Optional[str] = 'sd_final_output',
        save_intermediate_latents: bool = True,
        display: bool = False
) -> int:
    """
    Stable Diffusion v2.1 Turbo onnxruntime pipeline
    """
    seq_session_options = ORTSessionOptions('CPU', 'cpu', {}, 2, 0, onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL,
                                            onnxruntime.ExecutionMode.ORT_SEQUENTIAL,
                                            onnxruntime.ExecutionOrder.DEFAULT, 3)
    
    acc_seq_session_options = ORTSessionOptions('NPU', 'cpu', {}, 2, 0, onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL,
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

    cond_text_embedding, text_encoder_1_inference_time, text_encoder_2_inference_time = runTextEncoder(
        model_directory,
        cond_input_ids,
        seq_session_options,
        False
    )

    if image is not None:
        latents, vae_encoder_inference_time = runVAEEncoder(
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

        vae_encoder_inference_time = 0.0
    
    # unet timestep embedding weights
    unet_time_embedding_weights = (
        numpy.load(os.path.join(weights_directory, 'sd_turbo_numpy', 'linear1Weight.npy')),
        numpy.load(os.path.join(weights_directory, 'sd_turbo_numpy', 'linear1Bias.npy')),
        numpy.load(os.path.join(weights_directory, 'sd_turbo_numpy', 'linear2Weight.npy')),
        numpy.load(os.path.join(weights_directory, 'sd_turbo_numpy', 'linear2Bias.npy'))
    )
    
    # De-Noising Loop
    print('\nImage de-noising loop...')
    for i, t in enumerate(tqdm(scheduler.timesteps)):
        batch_latents = scheduler.scale_model_input(torch.from_numpy(latents), timestep=t).detach().numpy()

        unet_output, unet_inference_time = runUNet(
            model_directory,
            batch_latents,
            getTimestepEmbedding(t.item(), *unet_time_embedding_weights),
            cond_text_embedding,
            seq_session_options,
            False
        )

        latents = scheduler.step(torch.from_numpy(unet_output), t, torch.from_numpy(latents)).prev_sample.detach().numpy()

        if save_intermediate_latents:
            # just to view intermediate latents
            int_latents = copy.deepcopy(latents)

            latent_norm_list.append((t.item(), numpy.linalg.norm(numpy.reshape(int_latents, (int_latents.shape[0], -1)), axis=1).mean()))

            _ = postProcess(model_directory, int_latents, seq_session_options, os.path.join(save_directory, 'latents'), 'sd_int_' + str(t.item()), False, False)

        # average inference time for unet model for conditional input
        if i > 0:
            unet_inference_time += unet_inference_time

    unet_inference_time = round(unet_inference_time / steps, 3)

    vae_decoder_inference_time = postProcess(model_directory, latents, seq_session_options, os.path.join(save_directory, 'images'), output_filename, False, display)

    end = time.time() - start

    logging.info('----- Total Inference Time for {} steps: {} s -----\n'.format(steps, end))

    # inference times in ms
    inference_times += (text_encoder_1_inference_time, text_encoder_2_inference_time, vae_encoder_inference_time, unet_inference_time, vae_decoder_inference_time, round(end * 1000, 3))

    dumpMetadata(
        inference_times,
        latent_norm_list,
        os.path.join(save_directory, 'logs'),
        "Text Encoder 1, Text Encoder 2, VAE Encoder, UNet, VAE Decoder, Total",
        filename='sd_pipeline'
    )

    return 0
