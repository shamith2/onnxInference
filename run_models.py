from onnx_helper import ONNXInference

import gc
import argparse
import itertools
import os
import re
import shutil
import sys
import time
from pathlib import Path
from multiprocessing import Process, Queue
import subprocess
import winreg
import requests

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__producer__ = "run_models"
__version__ = "1.0.0"


def preprocess_files(models_dir, workspace, model_type, args=None):
    if os.path.exists(os.path.join(workspace, model_type + '.txt')):
        return 0

    if model_type == 'procyon-v1.1' or model_type == 'procyon-v1.1-rc2':
        model_names = []
        models_path = []
        for root, dirs, files in os.walk(models_dir, topdown=True):
            for m_dir in dirs:
                if '_tf' in m_dir:
                    model_names.append(m_dir[:-3].lower())
                else:
                    model_names.append(m_dir.lower())

            for f in files:
                if 'qdq' not in f:
                    models_path.append(os.path.join(root, f))

        assert len(model_names) == len(models_path)

        for im, mn_path in enumerate(models_path):
            p_ort_inf = ONNXInference(model_name=model_names[im], metadata='test', model_path=mn_path, mode='fp32')

            # skip esrgan model
            if 'esrgan' not in model_names[im]:
                _ = p_ort_inf.quantize()

                time.sleep(2)

                for (l, cf) in itertools.product(('1x4', '4x4'), ('vaip_config', 'vaip_config_opt')):
                    _p = Process(target=p_ort_inf.start_inference, args=(1,
                                                                         l,
                                                                         cf,
                                                                         True,
                                                                         1,
                                                                         False,
                                                                         False,
                                                                         False,
                                                                         None,
                                                                         None,
                                                                         1,
                                                                         'ryzen-ai',
                                                                         3,
                                                                         1))

                    _p.start()

                    _p.join()

                    _p.terminate()

                    _p.close()

                    time.sleep(5)

        # skip esrgan model
        try:
            model_names.remove('esrgan')
        except Exception as _:
            pass

        with open(os.path.join(workspace, model_type + '.txt'), 'w') as f:
            for mn in model_names:
                f.write(mn + '\n')

    else:
        pass

    return 0


# adapted from https://www.geeksforgeeks.org/manipulating-windows-registry-using-winreg-in-python/
# and https://amgeneral.wordpress.com/2021/04/19/battery-power-slider-aka-windows-performance-power-slider-registry-key/
def changeRegistry(slider_index: int = None, mode: str = 'ac', action: str = 'read'):
    if slider_index is None and mode != 'npu' and action != 'read':
        raise Exception("slider_index cannot be None when action is not read and mode is not npu")

        return 2

    if mode not in ['ac', 'dc', 'npu']:
        raise Exception("Check mode value")

        return 3

    options = {'BestEfficiency': '961cc777-2547-4f9d-8174-7d86181b8a7a',
               'Balanced': '3af9B8d9-7c97-431d-ad78-34a8bfea439f',
               'BestPerformance': 'ded574b5-45a0-4f42-8737-46345c09c238'}

    if mode == 'npu':
        slider_position = None
    else:
        slider_position = list(options.keys())[int(slider_index) - 1]

    # Store location of HKEY_LOCAL_MACHINE
    location = winreg.HKEY_LOCAL_MACHINE

    # Storing path in soft
    if action == 'read':
        if mode != 'npu':
            registry = winreg.OpenKeyEx(location, r"SYSTEM\\CurrentControlSet\\Control\\Power\\User\\PowerSchemes", 0,
                                        winreg.KEY_QUERY_VALUE)

        else:
            registry = winreg.OpenKeyEx(location, r"SYSTEM\\ControlSet001\\Services\\kipudrv\\Parameters", 0,
                                        winreg.KEY_QUERY_VALUE)

    else:
        try:
            registry = winreg.OpenKeyEx(location, r"SYSTEM\\ControlSet001\\Services\\kipudrv\\Parameters", 0,
                                        winreg.KEY_QUERY_VALUE + winreg.KEY_SET_VALUE)
        except PermissionError:
            logging.error(
                'Change permissions for HKLM\\SYSTEM\\ControlSet001\\Services\\kipudrv\\Parameters for USERS to Full Control. ' +
                'Owner can be SYSTEM. Disable Inheritance not required. Refer to https://www.groovypost.com/howto/disable-modify-uac-user-account-control-notification-windows/\n')

            return 4

    if mode == 'ac':
        key = 'ActiveOverlayAcPowerScheme'
    elif mode == 'dc':
        key = 'ActiveOverlayDcPowerScheme'
    else:
        key = 'FirmwareControlOption'

    if action == 'read':
        # reading value in registry key
        value = winreg.QueryValueEx(registry, key)[0]

        if mode != 'npu':
            slier_pos = None
            for i, val in enumerate(options.values()):
                if val == value:
                    slider_pos = list(options.keys())[i]

            logging.info("Power Slider: {}\n".format(slider_pos))

        else:
            logging.info("FirmwareControlOption: {}\n".format(hex(value)))

    else:
        if mode != 'npu':
            try:
                value = winreg.QueryValueEx(registry, 'FirmwareControlOption')[0]

                if value:
                    winreg.DeleteValue(registry, 'FirmwareControlOption')
            except FileNotFoundError:
                pass

            ps_ret = subprocess.run([".\PowerMode.exe", slider_position], shell=True, capture_output=True)

            if not ps_ret.returncode:
                logging.info("Changed Power Mode successfully!!\n")

        else:
            try:
                value = winreg.QueryValueEx(registry, key)[0]

                if hex(value) != 0x2:
                    winreg.SetValueEx(registry, key, 0, winreg.REG_DWORD, 0x2)

                else:
                    if registry:
                        winreg.CloseKey(registry)

                    return 1

            except FileNotFoundError as e:
                winreg.SetValueEx(registry, key, 0, winreg.REG_DWORD, 0x2)

    if registry:
        winreg.CloseKey(registry)

    return 0


def agm_start(outfile, interval=1000):
    agm_start_command = (
            'START /d "C:\\Program Files\\AMD Graphics Manager" /min AMDGraphicsManager.exe -minimize -ignorewarnings -unilogallgroups -unilog=pm -unilogperiod='
            + str(interval) + ' -unilogoutput=' + str(outfile) + ' -unilogstopcheck')

    time.sleep(1)

    os.system(agm_start_command)

    time.sleep(1)


def wpr_start():
    wpr_start_command = 'wpr -start power -filemode'

    time.sleep(1)

    os.system(wpr_start_command)

    time.sleep(1)


def agm_stop():
    agm_stop_command = 'echo > "C:\\Program Files\\AMD Graphics Manager\\terminate.txt"'

    time.sleep(1)

    os.system(agm_stop_command)

    time.sleep(1)


def wpr_stop(outfile):
    wpr_stop_command = 'wpr -stop ' + outfile

    time.sleep(1)

    os.system(wpr_stop_command)

    time.sleep(1)


def rename_files(working_dir, models, log_cat, metadata, mode='ryzen-ai'):
    logs_root = os.path.join(working_dir, 'onnx', log_cat, metadata)

    result_files = []
    log_files = []

    if mode == 'ryzen-ai':
        mode_dir = 'ryzen_ai'

    for _model in models:
        results_root = os.path.join(working_dir, 'onnx', 'results', mode_dir, _model, metadata)

        for r_f in os.listdir(results_root):
            result_files.append(os.path.join(results_root, r_f))

    for l_f in os.listdir(logs_root):
        log_files.append(os.path.join(logs_root, l_f))

    assert len(result_files) == len(log_files)

    for i in range(len(result_files)):
        fps = result_files[i].split('_')[-2]

        old_name_list = log_files[i].split('_')
        old_name_list.insert(-1, fps)
        new_name = '_'.join(old_name_list)

        os.rename(log_files[i], new_name)

    return 0


if __name__ == '__main__':

    # tunable parameters

    parser = argparse.ArgumentParser(
        description='ONNX Benchmarking on Strix IPU/NPU. Use default values for optimal performance.')
    parser.add_argument('--models', type=str, nargs='+', action='store',
                        help='names of models to run inference on. names should match the name each model was quantized as. if --quantize is used, models will be quntized under the given names. ' +
                             'default: --models deeplabv3 inceptionv4 mobilenetv3 resnet50 yolov3',
                        default=['deeplabv3', 'inceptionv4', 'mobilenetv3', 'resnet50', 'yolov3'])
    parser.add_argument('--model_type', type=str, action='store',
                        help='type of models. supported: procyon-v1.1, procyon-v1.1-rc2. default: --model_type procyon-v1.1-rc2',
                        default='procyon-v1.1-rc2')
    parser.add_argument('--num_input', '-n', type=int, action='store',
                        help='number of images/inputs to run inference on. default: --num_input 100', default=100)
    parser.add_argument('--instance_count', '-i', type=int, nargs='+', action='store',
                        help='number of npu columns to run inference on. per config. ' +
                             'to run benchmarks for different number of instance count, pass values for each config, like --instance_count 1 8. ' +
                             'use more than the total npu columns available to reduce npu starving. default: --instance_count 1',
                        default=[1])
    parser.add_argument('--time_limit', '-t', type=int, action='store',
                        help='how long (in sec) to run benchmarks for each model. default: --time_limit 120',
                        default=120)
    parser.add_argument('--num_threads', '-nt', type=int, nargs='+', action='store',
                        help='number of cpu threads to use to run inference. per config. ' +
                             'to run benchmarks for different number of cpu threads, pass values for each config, like --num_threads 8 16. default: --num_threads 8',
                        default=[8])
    parser.add_argument('--intra_threads', '-it', type=int, action='store',
                        help='number of onnxruntime intra threads to use. default: --intra_threads 1', default=1)
    parser.add_argument('--enable_thread_spinning', action='store_true',
                        help='whether to enable thread spinning', default=False)
    parser.add_argument('--config_file', '-vc', type=str, action='store',
                        help='which config file to use. for best peformance, use vaip_config_opt. default: --config_file vaip_config_opt',
                        default="vaip_config_opt")
    parser.add_argument('--layout', '-xb', type=str, action='store',
                        help='which xclbin to use. for 1x1x4 config, use "1x4" else for 4x4 config use "4x4". default: --layout 4x4',
                        default="4x4")
    parser.add_argument('--metadata', '-m', type=str, action='store',
                        help='metadata information to use in the file name when saving results and agm logs. default: --metadata test',
                        default="test")
    parser.add_argument('--rename_files', '-rf', action='store_true',
                        help='whether to rename log files', default=False)
    parser.add_argument('--summarize', '-sf', action='store_true',
                        help='whether to summarize log files', default=False)
    parser.add_argument('--compile_only', '-co', action='store_true',
                        help='if this option is used, compile the models and store them in cache. does not run inference when this option is used',
                        default=False)
    parser.add_argument('--quantize', '-qc', action='store_true',
                        help='if this option is used, quantize, compile the models and store them in cache. does not run inference when this option is used',
                        default=False)
    parser.add_argument('--profiling', '-prof', action='store_true',
                        help='for profiling',
                        default=False)
    parser.add_argument('--agm', action='store_true',
                        help='if this option is used, collect agm log',
                        default=False)
    parser.add_argument('--etl', action='store_true',
                        help='if this option is used, collect etl trace',
                        default=False)
    parser.add_argument('--registry', type=str, action='store',
                        help='which registry keys to change. to change ac power slider to best performance, use --registry ac3. ' +
                             'to set npu driver registry key for manual dpm, use --registry npu0. ' +
                             'will fail if user does not have correct permissions.',
                        default=None)

    # tunable parameters
    args = parser.parse_args()

    if args.registry:
        args.registry = args.registry + '0' if args.registry == 'npu' else args.registry

        rtv = changeRegistry(int(args.registry[-1]), mode=args.registry[:-1], action='write')

        if not rtv:
            if 'npu' in args.registry:
                os.system('shutdown /r')
                sys.exit()

            else:
                time.sleep(120)

        elif rtv == 1 and 'npu' in args.registry:
            sys.exit()

        else:
            pass

    kwargs = {}

    kwargs['working_dir'] = Path(__file__).parent.resolve()
    kwargs['model_name'] = None
    kwargs['instance_count'] = None
    kwargs['num_threads'] = None

    os.chdir(kwargs['working_dir'])

    if args.rename_files:
        rename_files(kwargs['working_dir'], args.models, 'agm_logs', args.metadata)
        sys.exit()

    if args.summarize:
        agm_dir = os.path.join(kwargs['working_dir'], 'onnx', 'agm_logs', args.metadata)
        subprocess.run(['python', 'parse_agm.py', args.metadata, agm_dir, agm_dir], shell=True, capture_output=False)
        sys.exit()

    if args.model_type not in ['procyon-v1.1', 'procyon-v1.1-rc2']:
        raise Exception("This framework currently only supports procyon-v1.1, procyon-v1.1-rc2")

    if args.model_type == 'procyon-v1.1':
        preprocess_files(os.path.join(kwargs['working_dir'], 'procyon-v1.1'), kwargs['working_dir'], args.model_type)
    elif args.model_type == 'procyon-v1.1-rc2':
        preprocess_files(os.path.join(kwargs['working_dir'], 'procyon-v1.1-rc2'), kwargs['working_dir'],
                         args.model_type)
    else:
        pass
    
    if args.quantize:
        sys.exit()

    models = args.models
    for model_dir in args.models:
        if len(os.listdir(os.path.join(kwargs['working_dir'], 'onnx', 'onnx_models', 'ryzen_ai', model_dir))) == 0:
            models.remove(model_dir)

    # run standalone onnx models
    for (isc, nt) in list(itertools.product(args.instance_count, args.num_threads)):
        kwargs['instance_count'] = isc
        kwargs['num_threads'] = nt

        for i, model_name in enumerate(models):
            logging.info("Setting system idle for 2 minutes...\n")
            time.sleep(120)

            kwargs['model_name'] = model_name

            ort_inf = ONNXInference(model_name=model_name, metadata=args.metadata)

            if args.agm:
                agm_dir = os.path.join(kwargs['working_dir'], 'onnx', 'agm_logs', args.metadata)
                os.makedirs(agm_dir, exist_ok=True)

                agm_outfile = "{}_{}_onnx_{}_{}n_{}t_{}i_{}nt_{}it_{}dts_run1.csv".format(kwargs['model_name'],
                                                                                          args.metadata,
                                                                                          args.config_file,
                                                                                          args.num_input,
                                                                                          args.time_limit,
                                                                                          kwargs['instance_count'],
                                                                                          kwargs['num_threads'],
                                                                                          args.intra_threads,
                                                                                          str(int(
                                                                                              not args.enable_thread_spinning)))

            if args.etl:
                wpr_dir = os.path.join(kwargs['working_dir'], 'onnx', 'etl_traces', args.metadata)
                os.makedirs(wpr_dir, exist_ok=True)

                wpr_outfile = "{}_{}_onnx_{}_{}n_{}t_{}i_{}nt_{}it_{}dts_run1.etl".format(kwargs['model_name'],
                                                                                          args.metadata,
                                                                                          args.config_file,
                                                                                          args.num_input,
                                                                                          args.time_limit,
                                                                                          kwargs['instance_count'],
                                                                                          kwargs['num_threads'],
                                                                                          args.intra_threads,
                                                                                          str(int(
                                                                                              not args.enable_thread_spinning)))

            if args.etl:
                pass  # wpr_start()

            if args.agm:
                agm_start(os.path.join(agm_dir, agm_outfile))

            p = Process(target=ort_inf.start_inference, args=(kwargs['instance_count'],
                                                              args.layout,
                                                              args.config_file,
                                                              args.compile_only,
                                                              args.num_input,
                                                              True,
                                                              args.profiling,
                                                              not args.enable_thread_spinning,
                                                              args.time_limit,
                                                              0,
                                                              kwargs['num_threads'],
                                                              'ryzen-ai',
                                                              3,
                                                              args.intra_threads))

            p.start()

            p.join()

            p.terminate()

            p.close()

            if args.etl:
                pass  # wpr_stop(os.path.join(wpr_dir, wpr_oufile))

            if args.agm:
                agm_stop()
