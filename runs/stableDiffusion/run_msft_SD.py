# Script to run Stable Diffusion pipelines

import os
from pathlib import Path

from tqdm.auto import tqdm

import numpy
from PIL import Image

from onnxInsights.stableDiffusion import SD_Turbo_MSFT_pipeline


# global variables: use with caution

ROOT = Path(__file__).parents[2].resolve()
WORKSPACE = Path(__file__).parent.resolve()

MODEL_DIR = os.path.join(ROOT, 'weights', 'stableDiffusion', 'sd_msft_onnx')

WEIGHTS_DIR = os.path.join(ROOT, 'weights', 'stableDiffusion')

DATA_DIR = os.path.join(ROOT, 'data', 'stableDiffusion')

RESULT_DIR = os.path.join(ROOT, 'results', 'stableDiffusion', 'sd_msft_results')


# runs

def run_sd_turbo_text2img():
    prompt = "An astronaut riding a horse"
    
    SD_Turbo_MSFT_pipeline(prompt, MODEL_DIR, WEIGHTS_DIR, None, steps=4, save_directory=RESULT_DIR, output_filename = 'image', display=False)


if __name__ == '__main__':
    run_sd_turbo_text2img()

