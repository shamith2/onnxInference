# Helper functions for Python ONNXRuntime Inference Pipelines

import os
import numpy
import random
from datetime import datetime

import torch

from PIL import Image, ImageOps
from matplotlib import pyplot as plt

import cv2
import pyautogui

from typing import Optional

import imagehash

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__producer__ = "inferenceHelper"
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
) -> Image:
    """
    Capture Screenshot

    inputs:
        None

    outputs:
        return 0 if Success
    """
    image = pyautogui.screenshot()

    return image


def saveScreenshot(
        image: Image,
        save_directory: str
) -> int:
    """
    Save Screenshot

    inputs:
        filename: str = save the screenshot as
        save_directory: str = save the screenshot as in

    outputs:
        return screenshot if Success
    """
    date, time = datetime.now().strftime("%Y%m%d %H%M%S").split(' ')

    screenshot_directory = os.path.join(save_directory, date)

    if not os.path.exists(screenshot_directory):
        os.makedirs(screenshot_directory)

    filename = os.path.join(screenshot_directory, time + '.png')

    image.save(filename)

    return 0


def compareScreenshots(
        screenshot1: Image,
        screenshot2: Image,
        hamming_limit: int
) -> bool:
    """
    outputs:
        return True if screenshots are similar else False
    """
    hash1 = imagehash.phash(screenshot1)
    hash2 = imagehash.phash(screenshot2)

    hamming_distance = hash2 - hash1

    if hamming_distance <= hamming_limit:
        return True
    
    else:
        print('hamming distance: ' + str(hamming_distance))
        return False


def embeddingCosineSimilarity(embedding1, embedding2):
    normalized_embedding1 = numpy.linalg.norm(embedding1)
    normalized_embedding2 = numpy.linalg.norm(embedding1)

    if normalized_embedding1 == 0 or normalized_embedding2 == 0:
        return 0
    
    else:
        return numpy.dot(embedding1, embedding2) / (normalized_embedding1 * normalized_embedding2)


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    
    return next_token


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
    
    if latent_norm_list:
        for t, latents_norm in latent_norm_list:
            with open(os.path.join(result_dir, latent_norm_filename), 'a') as f:
                f.write("Step: {} Latenct Norm: {}\n".format(t, latents_norm))
    
    return 0


def siLU(
        inputTensor: numpy.ndarray
) -> numpy.ndarray:
    return inputTensor / (1 + numpy.exp(-inputTensor))


def softmax(
        inputTensor: numpy.ndarray
) -> numpy.ndarray:
    return numpy.exp(inputTensor) / numpy.exp(inputTensor).sum()


def preProcessTensor(
    image: numpy.ndarray,
    preprocess: bool = True
) -> numpy.ndarray:
    tensor = changeDtype(image / 255.0, "tensor(float)")

    if preprocess:
        tensor = tensor * 2.0 - 1

    c1, c2, c3 = tensor.shape

    if c3 == 3 and (c2 > c3 and c1 > c3):
        tensor = numpy.transpose(tensor, (2, 0, 1))

    tensor = numpy.expand_dims(tensor, axis=0)

    return tensor


def getTensorfromImage(
        image_path: str,
        image_size: int = None,
        preprocess: bool = True
) -> numpy.ndarray:
    image = Image.open(image_path)

    image = image.convert('RGB')

    if image_size:
        image = image.resize((image_size, image_size))

    tensor = preProcessTensor(numpy.array(image), preprocess=preprocess)

    return tensor


def hdTransform(
        image_path: str,
        max_num_crops: int,
        image_size: int
) -> tuple[numpy.ndarray, numpy.ndarray, int]:
    img = Image.open(image_path)

    image_mean = numpy.array([0.48145466, 0.4578275, 0.40821073])
    image_std = numpy.array([0.26862954, 0.26130258, 0.27577711])

    img = img.convert('RGB')
    w, h = img.size

    trans = False
    
    if w < h:
        img = img.transpose(Image.TRANSPOSE)
        trans = True
        w, h = img.size
    
    scale = int(numpy.sqrt(max_num_crops * w / h))
    img = img.resize([int(scale * image_size), int(scale * image_size * h / w)], Image.BILINEAR)
    
    def _pad(b):
        _, h = b.size
        diff_height = int(numpy.ceil(h / image_size) * image_size) - h
        top_padding = int(diff_height / 2)
        bottom_padding = diff_height - top_padding
        
        b = ImageOps.expand(b, border=(0, top_padding, 0, bottom_padding), fill=(255, 255, 255))
        
        return b
    
    img = _pad(img)
    img = img.transpose(Image.TRANSPOSE) if trans else img

    img = ((numpy.array(img) / 255.0 - image_mean) / image_std).transpose(2, 0, 1)

    h, w = img.shape[1], img.shape[2]
    shapes = numpy.array([[h, w]], dtype=numpy.int64)

    num_img_tokens = int((h // image_size * w // image_size + 1) * 144 + 1 + (h // image_size + 1) * 12)
    
    global_image = torch.nn.functional.interpolate(torch.from_numpy(img[None]), size=(image_size, image_size), mode='bicubic').clamp(min=0, max=255).numpy()
    
    hd_image = numpy.reshape(img, (1, 3, h // image_size, image_size, w // image_size, image_size))

    hd_image = numpy.transpose(hd_image, (0, 2, 4, 1, 3, 5)).reshape(-1, 3, image_size, image_size)

    hd_image = numpy.concatenate([global_image, hd_image], axis=0)

    hd_image = _reshape_to_max_num_crops_tensor(hd_image, max_num_crops)
    
    return hd_image, shapes, num_img_tokens


def _reshape_to_max_num_crops_tensor(
        tensor: numpy.ndarray,
        max_crops: int
) -> numpy.ndarray:
    """
    tensor: (bsz, num_channels, height, width) where B <= max_crops -> (batch_size, max_crops, 3, height, width)
    """
    bsz, num_channels, height, width = tensor.shape
    
    if bsz < max_crops:
        pad = numpy.zeros((max_crops - bsz, num_channels, height, width))
        
        tensor = numpy.concatenate((tensor, pad), axis=0)
    
        tensor = numpy.expand_dims(tensor, axis=0)
    
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
