# Helper functions for Python Stable Diffusion ONNXRuntime Pipelines

import os
import numpy
import random

from PIL import Image
from matplotlib import pyplot as plt

import cv2
import pyautogui

from typing import Optional

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__producer__ = "sdHelper"
__version__ = "0.1.0"


def changeDtype(
        tensor: numpy.ndarray,
        in_type: str
) -> numpy.ndarray:
    """
    Infer the correct dtype for inputs and outputs
    
    inputs:
        tensor: numpy.ndarray = tensor to change dtype
        in_type: str = onnx model input dtype
    
    outputs:
        numpy.ndarray = tensor in the appropriate numpy dtype 
    """
    if in_type == "tensor(float16)":
        return tensor.astype(numpy.float16)

    elif in_type == "tensor(float)":
        return tensor.astype(numpy.float32)

    elif in_type == "tensor(long)" or in_type == "tensor(int64)":
        return tensor.astype(numpy.int64)
    
    elif in_type == "tensor(int32)":
        return tensor.astype(numpy.int32)

    elif in_type == "tensor(uint)":
        return tensor.astype(numpy.uint8)

    else:
        return tensor.astype(numpy.float64)


def captureScreenshot(
        width: int,
        height: int
) -> numpy.ndarray:
    """
    Capture Screenshot

    inputs:
        width: unsigned int = width of the screenshot
        height: unsigned int = height of the screenshot

    outputs:
        screenshot: numpy.ndarray = screenshot image, as numpy array, of shape (width, height, 3)
    """
    screenshot = pyautogui.screenshot()

    screenshot = screenshot.resize(size=(width, height))

    screenshot = changeDtype(screenshot, "tensor(float)")
    
    screenshot = screenshot / 255.0

    return screenshot


def dumpMetadata(
        inference_times: tuple[float],
        latent_norm_list: tuple[float],
        result_dir: str,
        inference_time_header: str = "UNet,VAE Decoder,Total",
        latent_norm_header: str = "Step,Latents Norm",
        filename: str = 'sd_pipeline'
) -> int:
    """
    Helper function to save inference time of models to csv file

    Inputs:
        inference_times: tuple
        filename: str
    
    Outputs:
        return value = int
    """
    inf_time_filename = filename + '.csv'
    latent_norm_filename = 'int_norm.csv'

    if not os.path.exists(os.path.join(result_dir, inf_time_filename)):
        with open(os.path.join(result_dir, inf_time_filename), 'w') as f:
            f.write(inference_time_header + '\n')
    
    with open(os.path.join(result_dir, latent_norm_filename), 'w') as f:
        f.write(latent_norm_header + '\n')
    
    with open(os.path.join(result_dir, inf_time_filename), 'a') as f:
        for i, inference_time in enumerate(inference_times):
            if i != len(inference_times) - 1:
                f.write(str(inference_time) + ',')
            
            else:
                f.write(str(inference_time))

        f.write('\n')
    
    for t, latents_norm in latent_norm_list:
        with open(os.path.join(result_dir, latent_norm_filename), 'a') as f:
            f.write("Step: {} Latenct Norm: {}\n".format(t, latents_norm))
    
    return 0


def siLU(
        inputTensor: numpy.ndarray
) -> numpy.ndarray:
    return inputTensor / (1 + numpy.exp(-inputTensor))


def preProcessTensor(
    image: numpy.ndarray
) -> numpy.ndarray:
    tensor = changeDtype(image / 255.0, "tensor(float)")

    tensor = tensor * 2.0 - 1

    c1, c2, c3 = tensor.shape

    if c3 == 3 and (c2 > c3 and c1 > c3):
        tensor = numpy.transpose(tensor, (2, 0, 1))

    tensor = numpy.expand_dims(tensor, axis=0)

    return tensor


def getTensorfromImage(
        image_path: str,
        image_size: int
) -> numpy.ndarray:
    image = Image.open(image_path)

    image = image.convert('RGB')

    image = image.resize((image_size, image_size))

    tensor = preProcessTensor(numpy.array(image))

    return tensor


# adpated from https://www.geeksforgeeks.org/python-program-extract-frames-using-opencv/
def getFramesfromVideo(
        video_path: str,
        image_size: int,
        frame_count: Optional[int] = None
) -> numpy.ndarray:
    # Path to video file 
    vidObj = cv2.VideoCapture(video_path)

    # keep count of the number of frames
    count = 0
  
    # checks whether frames were extracted 
    success = True

    # frames in video
    frames = []
  
    while success:
        # vidObj object calls read 
        # function extract frames 
        success, frame = vidObj.read()

        if success:
            frame = cv2.resize(frame, (image_size, image_size))

            frames.append(preProcessTensor(frame))

            count += 1
    
    frames = numpy.concatenate(frames, axis=0)

    if frame_count:
        start = random.randint(1, count - frame_count - 1)
        frames = frames[start:start + frame_count, :]
    
    return frames


# adapted from https://www.geeksforgeeks.org/python-create-video-using-multiple-images-using-opencv/
def createVideofromFrames(
        images_directory: str,
        filename: str,
        frame_rate: int = 1
) -> int:
    workspace = images_directory
    images_dir = os.path.join(workspace, 'images')
    
    frame_list = [os.path.join(images_dir, image) for image in os.listdir(images_dir) if image.endswith('.png')]

    frame = cv2.imread(frame_list[0])

    # setting the frame width, height width 
    # the width, height of first image 
    height, width, _ = frame.shape
  
    video = cv2.VideoWriter(os.path.join(images_directory, filename + '.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))
  
    # Appending the frames to the video one by one 
    for frame in frame_list: 
        video.write(cv2.imread(frame))
      
    # Deallocating memories taken for window creation
    cv2.destroyAllWindows()

    # releasing the video generated
    video.release()
    
    return 0


def saveTensorasImage(
        tensor: numpy.ndarray,
        result_dir: str,
        filename: str = 'sd',
        display: bool = False,
        normalize: bool = True
) -> int:
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if tensor.ndim == 4:
        tensor = numpy.squeeze(tensor, axis=0)

    c1, c2, c3 = tensor.shape

    if normalize:
        tensor = numpy.clip((tensor / 2) + 0.5, a_min=0, a_max=1)
    
    if c1 == 3 and (c2 > c1 and c3 > c1):
        tensor = numpy.transpose(tensor, (1, 2, 0))
    
    if display:
        plt.imshow(tensor)
        plt.show()

    im = Image.fromarray(changeDtype((tensor * 255).round(), "tensor(uint)"))
    im.save(os.path.join(result_dir, filename + '.png'))

    return 0


# Adapted from https://www.kaggle.com/code/deveshsurve/step-by-step-guide-to-implement-latent-diffusion?scriptVersionId=172213199&cellId=22
def visualizeLatents(
        workspace: str,
        title: str,
        prompt: str
):

    latents_dir = os.path.join(workspace, 'latents')

    image_files = [os.path.join(latents_dir, img) for img in os.listdir(latents_dir) if img.endswith('.png')]
    
    num_steps = len(image_files)

    t = numpy.linspace(0, 999, num_steps).round()[::-1].astype(numpy.int32)

    fig, _ = plt.subplots(1, num_steps, figsize=(20, 20))

    fig.suptitle(title + '\nPrompt: ' + prompt)
    
    for i, (ax, img_path) in enumerate(zip(fig.axes, sorted(image_files, reverse=True))):
        img = plt.imread(img_path)
        ax.imshow(img)

        ax.set_title('Generated Latent')
        ax.set_xlabel('timestep ' + str(t[i]))
    
    fig.tight_layout()

    path = os.path.join(workspace, title.replace(' ', '_') + '_visualize.png')
    
    fig.savefig(path)

    return 0
