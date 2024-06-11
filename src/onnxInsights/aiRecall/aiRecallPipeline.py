# Implement Python Custom AI Recall pipeline using ONNXRuntime

import os
import sys
import time
import collections
from pathlib import Path
import shutil

import multiprocessing
import threading

import keyboard

from typing import Union, Optional
from tqdm.auto import tqdm

import torch

import numpy
import onnxruntime
import onnxruntime_genai as og

from transformers import LlamaTokenizerFast

from ..onnxHelpers import ORTSessionOptions, init_Inference, Inference
from ..onnxHelpers import changeDtype, embeddingCosineSimilarity, sample_top_p, captureScreenshot, saveScreenshot, compareScreenshots, hdTransform

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__producer__ = "aiRecallPipeline"
__version__ = "0.1.0"


# global variables: use with caution
ROOT = Path(__file__).parents[3].resolve()
WORKSPACE = Path(__file__).parent.resolve()

CACHE_DIR = os.path.join(ROOT, 'weights', 'aiRecall', '.cache')

RESULT_DIR = os.path.join(ROOT, 'results', 'aiRecall')

SNAPSHOT_DIRECTORY = os.path.join(ROOT, 'results', 'aiRecall', 'snapshots')

CONTEXT_LENGTH = 4096
IMG_SIZE = 336
MAX_NUM_CROPS = 16
NUM_IMAGE_TOKENS = 144

CHUNK_SIZE = 512
OVERLAPPING_CHUNK_SIZE = 64

VISION_MAX_LENGTH = 3072
TEXT_MAX_LENGTH = 4096


TextModelOptions = collections.namedtuple(
    'TextModelOptions',
    'temperature top_p'
)


def analyseSnapshots(
        model_directory: str
) -> tuple[tuple[str, str]]:
    """
    Analyse and understand what an user is doing in a screenshot/snapshot
    """
    prompt = "Extract the most important information from what is happening in the image, include URLs if applicable"
    assistant_template = '<|user|>\n<|image_1|>\n' + prompt + '<|end|>\n<|assistant|>'

    if not os.path.exists(SNAPSHOT_DIRECTORY):
        os.makedirs(SNAPSHOT_DIRECTORY)

    snapshots = ()
    analysis_report = ()

    for root, _, files in os.walk(SNAPSHOT_DIRECTORY):
        for file in files:
            if file.endswith('.png') and file[:-4].isnumeric():
                snapshots += (os.path.join(root, file),)
    
    logging.info('analyseSnapshots:\n')

    for snapshot in tqdm(snapshots):
        output = runVisionModel(
            assistant_template,
            model_directory,
            snapshot
        )

        analysis_report += ((snapshot, output),)
    
    return analysis_report


def generateFilename(
        analysis: str,
        model_directory: str
):
    prompt = "Generate a short concise relevant filename for what is happening in the following description: " + analysis
    assistant_template = '<|user|>\n' + prompt + '<|end|>\n<|assistant|>'

    output = runTextModel(
        assistant_template,
        model_directory
    )

    output = Path(output.strip('"')).stem

    return output


def runTextModel(
        assistant_template: str,
        model_directory: str
):
    """
    Run Phi-3 language model using onnxruntime-genai
    Adapted from https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/phi3-qa.py
    """
    model = og.Model(model_directory)

    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()

    search_options = {}
    search_options['max_length'] = TEXT_MAX_LENGTH

    input_tokens = tokenizer.encode(assistant_template)

    params = og.GeneratorParams(model)
    params.set_search_options(**search_options)
    params.input_ids = input_tokens
    generator = og.Generator(model, params)

    output = ''

    while not generator.is_done():
        generator.compute_logits()
        generator.generate_next_token()

        new_token = generator.get_next_tokens()[0]

        output += tokenizer_stream.decode(new_token)

    # delete the generator to free the captured graph
    del generator

    # remove trailing or leading space
    return output.strip()


def runTextModel2(
        analysis: str,
        model_directory: str,
        tokenizer: LlamaTokenizerFast,
        model_options: TextModelOptions,
        session_options: ORTSessionOptions
):
    """
    Run Phi-3 language model using onnxruntime
    """
    seq_session_options = session_options._replace(dynamic_axes = {'batch_size': 1})

    ort_session = init_Inference(
        os.path.join(model_directory, 'phi3-mini-128k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx'),
        session_options=seq_session_options
    )

    prompt = "Generate a short concise relevant filename for what is happening in the following description: " + analysis
    text_input = '<|user|>\n' + prompt + '<|end|>\n<|assistant|>'

    output = ''
    end_text_generation = False

    input_tokens = tokenizer(text_input, padding="max_length", max_length=TEXT_MAX_LENGTH, truncation=True, return_tensors='np')
    
    logits = changeDtype(input_tokens.input_ids, "tensor(int64)")

    attention_mask = changeDtype(input_tokens.attention_mask, "tensor(int64)")

    eos_id = tokenizer(tokenizer.eos_token).input_ids[1]

    kv_cache = ()

    for i in range(32 * 2):
        kv_cache += (numpy.zeros(shape=(1, 32, TEXT_MAX_LENGTH, 96), dtype=numpy.float32),)

    while not end_text_generation:
        model_inputs = (logits, attention_mask)
        model_inputs += kv_cache

        # compute logits and cache key and values
        outputs, _ = Inference(
            ort_session,
            model_inputs=model_inputs,
            memory_device=session_options.memory_device,
            sustain_tensors=False
        )

        logits = outputs[0]

        if model_options.temperature > 0:
            probs = torch.softmax(torch.from_numpy(logits)[:, -1] / model_options.temperature, dim=-1)
            logits = sample_top_p(probs, model_options.top_p)
        
        else:
            logits = torch.argmax(torch.from_numpy(logits[:, -1]), dim=-1)

        logits = logits.detach().numpy()

        next_token = logits.item()

        if next_token == eos_id:
            end_text_generation = True

        next_token = tokenizer.decode(next_token)

        output += next_token
        
        attention_mask = numpy.array([[1]], dtype=numpy.int64)

        kv_cache = ()

        for i in range(1, 32 * 2, 2):
            kv_cache += (outputs[i],)
            kv_cache += (outputs[i + 1],)

    # remove trailing or leading space
    output = output.strip()

    raise NotImplementedError


def runVisionModel(
        assistant_template: str,
        model_directory: str,
        image_path: Optional[str] = None
) -> str:
    """
    Run Phi-3 vision model using onnxruntime-genai
    Adapted from https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/phi3v.py
    """
    model = og.Model(model_directory)
    processor = model.create_multimodal_processor()
    tokenizer_stream = processor.create_stream()

    if image_path:
        image = og.Images.open(image_path)
    
    else:
        image = None
    
    inputs = processor(assistant_template, images=image)

    params = og.GeneratorParams(model)
    params.set_inputs(inputs)
    params.set_search_options(max_length=VISION_MAX_LENGTH)

    generator = og.Generator(model, params)

    output = ''

    while not generator.is_done():
        generator.compute_logits()
        generator.generate_next_token()

        new_token = generator.get_next_tokens()[0]
        
        output += tokenizer_stream.decode(new_token)

    # delete the generator to free the captured graph
    del generator

    # remove </s> and trailing or leading space
    return output[:-4].strip()
    

def runImageEncoder(
    model_directory: str,
    image_path: str,
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
    inputTensor, image_shape, num_img_tokens = hdTransform(image_path, MAX_NUM_CROPS, IMG_SIZE)

    bsz = inputTensor.shape[0]

    seq_session_options = session_options._replace(dynamic_axes = {'num_images': bsz, 'max_num_crops': MAX_NUM_CROPS, 'height': IMG_SIZE, 'width': IMG_SIZE})

    ort_session = init_Inference(
        os.path.join(model_directory, 'phi-3-v-128k-instruct-vision.onnx'),
        session_options=seq_session_options
    )

    outputs, inference_time = Inference(
        ort_session,
        model_inputs=(changeDtype(inputTensor, "tensor(float)"), changeDtype(image_shape, "tensor(int64)")),
        memory_device=session_options.memory_device,
        sustain_tensors=sustain_tensors
    )

    inference_time = round(inference_time / 1e6, 3)

    raise NotImplementedError


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
    seq_session_options = session_options._replace(dynamic_axes = {'batch_size': 1})

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


def saveAnalysis(
        analysis_reports: tuple[tuple[str, str]],
        tokenizer: Union[str, LlamaTokenizerFast],
        model_directory: str,
        session_options: ORTSessionOptions
) -> int:
    """
    Chunking analysis information for Retrieval-Augmented Generation (RAG)
    """
    database = {}

    if isinstance(tokenizer, str):
        tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer)

    logging.info('saveAnalysis:\n')

    for analysis_report in tqdm(analysis_reports):
        filename, analysis = analysis_report

        identifier = '/'.join(os.path.split(filename))

        chunks = []
        
        words = analysis.strip().split(' ')
        chunk = ''
        overlapping_chunk = ''
        count = 0

        for i, word in enumerate(words):
            if word:
                chunk += word + ' '
                count += len(word)

                if count >= CHUNK_SIZE - OVERLAPPING_CHUNK_SIZE:
                    overlapping_chunk += word + ' '

                if count > CHUNK_SIZE or i == len(words) - 1:
                    chunks.append(chunk)
                    
                    chunk = overlapping_chunk
                    overlapping_chunk = ''
                    count = 0

        with open(os.path.join(SNAPSHOT_DIRECTORY, 'database_text.log'), 'a') as f:
            for chunk in chunks:
                f.write(identifier + ': ')
                f.write(chunk.strip() + '\n')
        
        partial_embedding_list = []

        for chunk in chunks:
            text_tokens = tokenizer(chunk, padding=True, truncation=True, return_tensors='np').input_ids
            
            embedding, _ = runTextEncoder(
                model_directory,
                text_tokens,
                session_options,
                False
            )

            partial_embedding_list.append(embedding.mean(axis=1).squeeze(axis=0))

        database[identifier] = partial_embedding_list
        
        database_file = os.path.join(SNAPSHOT_DIRECTORY, 'database_embedding.npz')

        if os.path.exists(database_file):
            existing_database = numpy.load(database_file)

            # for key, chunk_embeddings in existing_database.items():
            #     if key in database.keys():
            #         database[key] = chunk_embeddings + database[key]
                
            #     else:
            #         database.update(key=chunk_embeddings)

            database.update(existing_database)
        
        numpy.savez(database_file, **database)
    
    return 0


def findClosestFilenames(
        query: str,
        top_k: float,
        tokenizer: Union[str, LlamaTokenizerFast],
        model_directory: str,
        session_options: ORTSessionOptions,
        best_search_results: Optional[multiprocessing.Queue] = None
):  
    if isinstance(tokenizer, str):
        tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer)

    text_tokens = tokenizer(query, padding=True, truncation=True, return_tensors='np').input_ids
            
    query_embedding, _ = runTextEncoder(
        model_directory,
        text_tokens,
        session_options,
        False
    )

    database = numpy.load(os.path.join(SNAPSHOT_DIRECTORY, 'database_embedding.npz'))

    cosine_scores = {}
    
    for identifier, chunk_embeddings in database.items():
        for chunk_embedding in chunk_embeddings:
            cosine_scores[identifier] = embeddingCosineSimilarity(query_embedding.mean(axis=1).squeeze(axis=0), chunk_embedding)
    
    sorted_cosine_scores = sorted(cosine_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    top_search_results = tuple([(identifier, score) for identifier, score in sorted_cosine_scores])

    if best_search_results is not None:
        best_search_results.put(top_search_results)
    
    else:
        return top_search_results


def captureSnapshots(
        interval: int,
        hamming_limit: int
):
    """
    Captures screenshot every 'interval' seconds and saves it if the hamming distance
    between the previously captured screenshot and the current screentshot < 'hamming_limit'
    """
    previous_screenshot = captureScreenshot()

    saveScreenshot(previous_screenshot, SNAPSHOT_DIRECTORY)

    while True:
        time.sleep(interval)

        current_screenshot = captureScreenshot()

        if not compareScreenshots(previous_screenshot, current_screenshot, hamming_limit):
            saveScreenshot(current_screenshot, SNAPSHOT_DIRECTORY)


def analyse_and_savetoRAG(
        model_directory: str,
        tokenizer: Union[str, LlamaTokenizerFast],
        session_options: ORTSessionOptions
):
    if isinstance(tokenizer, str):
        tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer)

    intermediate_analysis_reports = analyseSnapshots(
        os.path.join(model_directory, 'vision', 'cpu-int4-rtn-block-32-acc-level-4')
    )

    if len(intermediate_analysis_reports):
        analysis_reports = ()

        logging.info('generateFilename:\n')

        for analysis_report in tqdm(intermediate_analysis_reports):
            full_filepath = analysis_report[0]
            file_directory = os.path.split(Path(full_filepath).parent.resolve())[-1]
            time_ext_identifier = os.path.split(full_filepath)[-1]

            filename = generateFilename(
                analysis_report[1],
                os.path.join(model_directory, 'text', 'cpu-int4-rtn-block-32-acc-level-4')
            )

            filename = os.path.join(file_directory, filename + '_' + time_ext_identifier)
            
            shutil.move(analysis_report[0], os.path.join(SNAPSHOT_DIRECTORY, filename))
            
            analysis_reports += ((filename, analysis_report[1]),)

        saveAnalysis(
            analysis_reports,
            tokenizer,
            os.path.join(model_directory, 'vision', 'cpu-int4-rtn-block-32-acc-level-4'),
            session_options
        )


def AI_Recall_pipeline(
        model_directory: str,
        query_or_screenshot: str,
        top_p: int,
        save_directory: str = RESULT_DIR
) -> int:
    """
    Custom AI Recall pipeline using onnxruntime
    """
    seq_session_options = ORTSessionOptions('CPU', 'cpu', {}, 2, 0, onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL,
                                            onnxruntime.ExecutionMode.ORT_SEQUENTIAL,
                                            onnxruntime.ExecutionOrder.DEFAULT, 3)
    
    for dir in ['logs', 'images', 'latents']:
        path = os.path.join(save_directory, dir)
        
        if not os.path.exists(path):
            os.makedirs(path)

    start = time.time()

    screenshot_process = multiprocessing.Process(target=captureSnapshots, args=(5, 12))

    if query_or_screenshot is None:
        screenshot_process.start()

        print("\nEnter 'q' to exit...\n")

        while screenshot_process.is_alive():
            if keyboard.is_pressed('q'):
                screenshot_process.terminate()
                sys.exit()
    
    else:
        # initialization
        tokenizer = LlamaTokenizerFast.from_pretrained(os.path.join(model_directory, 'text', 'cpu-int4-rtn-block-32-acc-level-4'))

        query = query_or_screenshot

        analyse_and_savetoRAG(
            model_directory,
            tokenizer,
            seq_session_options
        )

        top_p_results = findClosestFilenames(
            query,
            top_p,
            tokenizer,
            os.path.join(model_directory, 'vision', 'cpu-int4-rtn-block-32-acc-level-4'),
            seq_session_options,
            None
        )

        end = time.time() - start

        logging.info('----- Top {} Search Results for Query "{}": -----\n'.format(top_p, query))

        for filename, score in top_p_results:
            logging.info('{}: {}\n'.format(filename, score))

        logging.info('----- Total Inference Time: {} s -----\n'.format(end))

    return 0

