# Script to test Stable Diffusion pipelines

import os
from pathlib import Path
import pytest

from tqdm.auto import tqdm

import numpy
from PIL import Image

from onnxInsights.stableDiffusion import SD_pipeline, SD_Turbo_pipeline, SDXL_Turbo_pipeline
from onnxInsights.stableDiffusion import getFramesfromVideo, createVideofromFrames, visualizeLatents


# global variables: use with caution

ROOT = Path(__file__).parents[2].resolve()
WORKSPACE = Path(__file__).parent.resolve()

SD_TURBO_MODEL_DIR = os.path.join(ROOT, 'weights', 'stableDiffusion', 'sd-turbo-onnx')
SD_XL_TURBO_MODEL_DIR = os.path.join(ROOT, 'weights', 'stableDiffusion', 'sdxl-turbo-fp16-onnx')

DATA_DIR = os.path.join(ROOT, 'data', 'stableDiffusion')

SD_TURBO_RESULT_DIR = os.path.join(ROOT, 'results', 'stableDiffusion', 'sd_turbo_results')
SD_XL_TURBO_RESULT_DIR = os.path.join(ROOT, 'results', 'stableDiffusion', 'sdxl_turbo_results')

img_prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"


# tests

def test_sd_turbo_text2img():
    prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe"
    
    assert SD_Turbo_pipeline(prompt, SD_TURBO_MODEL_DIR, None, steps=1, save_directory=SD_TURBO_RESULT_DIR, output_filename = 'image') == 0


def test_sd_turbo_text2img():
    img_prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"

    image = Image.open(os.path.join(DATA_DIR, 'cat.png')).convert('RGB').resize((512, 512))
    
    assert SD_Turbo_pipeline(img_prompt, SD_TURBO_MODEL_DIR, image=image, steps=1, save_directory=SD_TURBO_RESULT_DIR, output_filename = 'image') == 0


def test_sd_turbo_vid2vid():
    img_prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"

    buffer = getFramesfromVideo(os.path.join(DATA_DIR, 'dance.mp4'), 512, 2)
    count = buffer.shape[0]

    image = Image.open(os.path.join(DATA_DIR, 'cat.png')).convert('RGB').resize((512, 512))

    image_list = numpy.split(buffer, count, axis=0)

    for i, image in enumerate(tqdm(image_list)):
        assert SD_Turbo_pipeline(img_prompt, SD_TURBO_MODEL_DIR, image=image, steps=1, save_directory=os.path.join(DATA_DIR, 'video'), output_filename='image' + str(i)) == 0

    assert createVideofromFrames(os.path.join(DATA_DIR, 'video'), filename='sd', frame_rate=1) == 0


def test_sd_turbo_img2img():
    prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe"

    assert SDXL_Turbo_pipeline(prompt, SD_XL_TURBO_MODEL_DIR, steps=1, save_directory=SD_XL_TURBO_RESULT_DIR, output_filename = 'image') == 0


def test_visualize_latents():
    img_prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"

    assert visualizeLatents(SD_TURBO_RESULT_DIR, 'SD 2.1 Turbo', img_prompt) == 0

