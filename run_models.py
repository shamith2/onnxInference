# script for running benchmarking onnx models on ryzen ai
# only use this script as reference

from onnx_helper import *

import os
import time
from pathlib import Path
from multiprocessing import Process

os.chdir(os.path.join(Path(__file__).parent.resolve())

models = os.listdir('onnx_models')

models_path = [os.path.join(Path(__file__).parent.resolve(), 'onnx_models', model_name, model_name + '_fp32.onnx') for model_name in models]

if __name__ == '__main__':

    args = {}

    # tunable parameters
    args['num_input'] = 100
    args['instance_count'] = 8
    args['time_limit'] = 180
    args['num_threads'] = 8
    args['cclk'] = 'mission'
    args['intra_threads'] = 1
    args['config_file'] = r'vaip_config.json'
    args['layout'] = '1x4'
    args['model_name'] = None
    args['metadata'] = None

    for (isc, nt) in [(8, 8)]:
        args['instance_count'] = isc
        args['num_threads'] = nt

        for i, model_name in enumerate(models):

            args['model_name'] = model_name

            ort_inf = ONNXInference(model_name=model_name, in_dir=False)

            # retval = ort_inf.quantize(models_path[i])
            # sys.exit()

            p = Process(target=ort_inf.start_inference, args=(args['instance_count'],
                                                              args['layout'],
                                                              args['config_file'],
                                                              args['num_input'],
                                                              True,
                                                              False,
                                                              False,
                                                              args['time_limit'],
                                                              0,
                                                              args['num_threads'],
                                                              'ryzen_ai',
                                                              3,
                                                              args['intra_threads']))

            p.start()

            p.join()

            p.terminate()

            p.close()

            time.sleep(120)
