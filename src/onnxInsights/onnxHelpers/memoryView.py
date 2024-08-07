# Local Memory and Cache View

import copy
import os
from pathlib import Path
from typing import Any

import numpy
import pandas
from matplotlib import pyplot as plt

import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__producer__ = "memoryView"
__version__ = "0.2.4"


# memory view
class memoryView:
    def __init__(
            self,
            model_dir: str,
            model_profile: str,
            outputs_profile: str
    ):
        self.root = Path(__file__).parents[3].resolve()
        self.workspace = Path(__file__).parent.resolve()

        model_dir = '_'.join(model_dir.split(' ')).lower()
        self.model_dataframe_file = model_profile
        self.outputs_database_file = outputs_profile

        self.prof_directory = os.path.join(self.root, 'results', 'onnxProfile')
        self.profile_logs_directory = os.path.join(self.prof_directory, 'logs', model_dir)
        self.mem_view_logs_directory = os.path.join(self.prof_directory, 'logs', model_dir,
                                                    'memoryView')

        for p in [self.prof_directory, self.mem_view_logs_directory]:
            if not os.path.exists(p):
                os.makedirs(p)

        self.log_files = {
            'memory_view': os.path.join(self.mem_view_logs_directory, 'memory_view.json'),
            'cache_view': os.path.join(self.mem_view_logs_directory, 'cache_view.json'),
            'main_memory_context': os.path.join(self.mem_view_logs_directory,
                                                'main_memory_context.json')
        }

        # hyperparameters
        self.frequency_threshold = 1
        self.imm_cachability_threshold = 1
        self.rounding_decimal = 6


    def reset(
            self
    ) -> None:
        self.log_memory_view = []
        self.log_memory_usage = []
        self.log_cache_view = []
        self.log_cache_memory_usage = []

        self.model_profile = pandas.read_csv(os.path.join(self.profile_logs_directory, self.model_dataframe_file))
        self.outputs_profile = pandas.read_csv(os.path.join(self.profile_logs_directory, self.outputs_database_file))

        # drop last row since it contains totals
        self.model_profile.drop(self.model_profile.tail(1).index, inplace=True)

        self.cache_occupied = 0.0
        self.memory_occupied = 0.0

        self.cache_parentKey = 'entries'
        self.cache_context = {self.cache_parentKey: {}}

        self.outputs_sequence = None


    def updateDict(
            self,
            dictionary: dict,
            subdict: str,
            key: str,
            value: Any,
            overwrite: bool = True,
            add: bool = False
    ) -> dict:
        if subdict:
            if dictionary.get(subdict, None) is None:
                dictionary[subdict] = {}

            _dict = dictionary[subdict]

        else:
            _dict = dictionary

        if _dict.get(key, None) is None or overwrite:
            _dict[key] = value

        else:
            if isinstance(_dict[key], tuple):
                _dict[key] += (value,)

            else:
                _dict[key] = (_dict[key], value)

        if add:
            _dict[key] = round(sum(_dict[key]), self.rounding_decimal)

        if subdict:
            dictionary[subdict] = _dict

        else:
            dictionary = _dict

        return dictionary


    def checkKeyinDict(
            self,
            dictionary: dict,
            key: str
    ) -> bool:
        return True if key in list(dictionary.keys()) else False


    def evaluateOutput(
            self,
            frequency: int,
            imm_cachability: int
    ) -> bool:
        if self.frequency_threshold > 1 or self.imm_cachability_threshold > 1:
            logging.warning('frequency threshold or imm_cachablity threshold is greater than 1. check if it is intended')

        if ((frequency <= self.frequency_threshold) and 
            (imm_cachability <= self.imm_cachability_threshold)):
            return False

        else:
            return True


    def evictKeyfromCache(
            self,
            key: str,
            memory: float
    ) -> None:
        del self.cache_context[self.cache_parentKey][key]
        self.cache_occupied -= memory

        self.refreshCache()


    def refreshCache(
            self
    ) -> None:
        # sort output_priority dict by frequency and imm_cachability
        # output with high imm_cachability implies that the output
        # needs to wait longer before it's used but high frequency
        # implies the output is used frequently
        self.cache_context[self.cache_parentKey] = dict(sorted(self.cache_context[self.cache_parentKey].items(),
                                         key=lambda x: (x[1]['frequency'], -1 * x[1]['imm_cachability'], x[1]['memory']),
                                         reverse=True))


    def logData(
            self,
            filename: str,
            log_data: list
    ) -> None:
        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=4, separators=(',',': '))


    def findOutliers(
            self,
            data: list,
            min_val: float,
            max_val: float
    ) -> list:
        filtered_data = []

        for memory_val in data:
            if memory_val >= min_val and memory_val <= max_val:
                filtered_data.append(memory_val)
        
        return filtered_data


    def plotMemory(
            self,
            memory_usage: list,
            min_val: float,
            max_val: float,
            steps: int,
            for_cache: bool = False,
            display: bool = False
    ) -> None:
        if not for_cache:
            if not min_val:
                min_val = min(memory_usage)

            if not max_val:
                max_val = max(memory_usage)

            filtered_mem_usage = self.findOutliers(
                memory_usage,
                min_val,
                max_val
            )

        else:
            filtered_mem_usage = memory_usage

        fig, ax = plt.subplots(figsize=(18, 12))

        x = range(len(filtered_mem_usage))
        min_val = min(filtered_mem_usage)
        max_val = max(filtered_mem_usage)

        _ = ax.scatter(
            x,
            filtered_mem_usage
        )

        # set axes labels and title
        ax.set_xticks(x)
        y_range = (numpy.linspace(min_val, max_val, steps, endpoint=True) if steps
                   else numpy.linspace(min_val, max_val, endpoint=True))
        ax.set_yticks(y_range)

        if not for_cache:
            ax.set_xlabel('Operator')      
            ax.set_ylabel('Total Memory Size [in MB]')

            fig.suptitle('Memory Profile\n', fontweight='bold')
            ax.set_title('Memory Size of Operators (>= {} MB and < {} MB)\n'.format(int(min_val), int(max_val) + 1) +
                        'Total Memory = Inputs Memory + Weights Memory + Outputs Memory')

        else:
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Memory used [in MB]')

            fig.suptitle('Cache Profile\n', fontweight='bold')
            ax.set_title('Cache Size (>= {} MB and < {} MB)\n'.format(int(min_val), int(max_val) + 1) +
                         'Cache Memory = Outputs Memory')

        plt.tick_params(bottom=True, labelbottom=False)
        plt.tight_layout()

        if display:
            plt.show()

        else:
            plt.close(fig)


    def updateCache(
            self,
            key: str,
            value: tuple[int, int, int, int, int, float]
    ) -> int:
        operator_id, output_id, next_input_id, frequency, imm_cachability, output_memory = value

        self.cache_occupied += output_memory

        # if output cannot fit in cache, evit least priority output
        # from cache which according to the sorting,
        # which is the last entry of the cache context
        while self.cache_occupied > self.cache_size:
            dict_keys, dict_values = zip(*self.cache_context[self.cache_parentKey].items())
            self.evictKeyfromCache(dict_keys[-1], dict_values[-1]['memory'])

        # add output to cache
        self.cache_context = self.updateDict(
            self.cache_context,
            subdict=self.cache_parentKey,
            key=key,
            value={
                'operator_id': operator_id,
                'output_id': output_id,
                'next_input_id': next_input_id,
                'frequency': frequency,
                'imm_cachability': imm_cachability,
                'memory': output_memory
            },
            overwrite=False
        )

        self.refreshCache()

        return 0


    def retrieveKeyfromCache(
            self,
            key: str
    ) -> int:
        output_dict = self.cache_context[self.cache_parentKey][key]

        operator_id = output_dict['operator_id']
        output_id = output_dict['output_id']
        frequency = output_dict['frequency']
        output_memory = output_dict['memory']

        output_idx = output_id - 1
        input_indices = self.outputs_profile['Input Node Index'].at[output_idx]

        # if the key is no longer used as input to other operators,
        # discard it
        if not isinstance(input_indices, str):
            self.evictKeyfromCache(key, output_memory)
            return 1

        # update input_indices for later use
        input_indices = [int(elem) for elem in input_indices.split(' ')]
        updated_input_indices = input_indices[1:]
        updated_input_indices = (' '.join([str(elem) for elem in updated_input_indices])
                                if updated_input_indices else numpy.nan)
        
        self.outputs_profile['Input Node Index'].at[output_idx] = updated_input_indices

        frequency -= 1
        next_input_id = input_indices[0]
        imm_cachability = next_input_id - output_dict['next_input_id']

        # evaluate if output is worth saving in the cache
        if self.evaluateOutput(frequency, imm_cachability):
            self.cache_context = self.updateDict(
                    self.cache_context,
                    subdict=self.cache_parentKey,
                    key=key,
                    value={
                        'operator_id': operator_id,
                        'output_id': output_id,
                        'next_input_id': next_input_id,
                        'frequency': frequency,
                        'imm_cachability': imm_cachability,
                        'memory': output_memory
                    },
                    overwrite=True
                )

            self.refreshCache()

        else:
            self.evictKeyfromCache(key, output_memory)
            return 2

        return 0


    def generate_view(
            self,
            memory_size: int,
            plot_memory: bool = False
    ) -> list[dict]:
        # memory size (in MB)
        self.memory_size = memory_size

        self.reset()

        operators_sequence = self.model_profile['Node']
        compute_operations_sequence = self.model_profile['Compute Operations']

        inputs_sequence = self.model_profile['Inputs Name']
        inputs_memory_seq = self.model_profile['Inputs Memory (in MB)']
        
        weights_sequence = self.model_profile['Weights and Bias Name']
        weights_memory_seq = self.model_profile['Weights and Bias Memory (in MB)']
        
        outputs_sequence = self.outputs_profile['Output Name']
        outputs_memory_seq = self.outputs_profile['Memory (in MB)']

        # assert sequence_length == len(outputs_sequence)
        sequence_length = len(operators_sequence)

        # operators are in sequence
        for operator_idx in range(sequence_length):
            self.memory_context = {}
            _no_weights = False

            # current operator to be executed
            _ = operators_sequence[operator_idx]

            # compute operations of operator (in GOPS)
            compute_ops = round(float(compute_operations_sequence[operator_idx]) / 1e6,
                                self.rounding_decimal)

            self.memory_context = self.updateDict(
                self.memory_context,
                subdict=None,
                key='id',
                value=(operator_idx + 1),
                overwrite=False
            )

            # inputs for operator
            op_inputs = inputs_sequence.at[operator_idx]
            inputs_memory = inputs_memory_seq.at[operator_idx]

            # weights for operator
            op_weights = weights_sequence.at[operator_idx]
            weights_memory = weights_memory_seq[operator_idx]

            if isinstance(op_inputs, str) and isinstance(inputs_memory, str):
                op_inputs = op_inputs.split(' ')
                inputs_memory = [float(elem) for elem in inputs_memory.split(' ')]

            else:
                op_inputs = [numpy.nan]
                inputs_memory = [0.0]

            if not isinstance(op_weights, str) or not weights_memory:
                _no_weights = True

            _input_metadata = [copy.deepcopy(op_inputs), copy.deepcopy(inputs_memory)]
            _prev_memory_adder = 0.0

            # check if inputs are outputs to previous operators,
            # yes, then, the operator can be fused provided
            # the memory size requirements are met
            if operator_idx > 0:
                output_list = self.log_memory_view[-1]['outputs'].keys()
                _prev_memory_adder = self.log_memory_view[-1]['total_memory']

                for i, op_input in enumerate(op_inputs):
                    if op_input and (op_input in output_list):
                        op_inputs[i] = numpy.nan
                        inputs_memory[i] = 0.0

            # current operator's output
            current_output = outputs_sequence[operator_idx]
            output_memory = outputs_memory_seq[operator_idx]

            inst_total_memory = round(sum(inputs_memory) + weights_memory + output_memory,
                                      self.rounding_decimal)

            if _prev_memory_adder + inst_total_memory < self.memory_size:
                if operator_idx > 0:
                    self.memory_context = self.log_memory_view[-1]

                self.memory_context = self.updateDict(
                    self.memory_context,
                    subdict=None,
                    key='id',
                    value=operator_idx + 1,
                    overwrite=False
                )

                # compute operators for operator
                self.memory_context = self.updateDict(
                    self.memory_context,
                    subdict=None,
                    key='compute',
                    value=compute_ops,
                    overwrite=False,
                    add=True
                )

                if _no_weights:
                    for i in range(len(op_inputs)):
                        if op_inputs[i] and inputs_memory[i]:
                            self.memory_context = self.updateDict(
                                self.memory_context,
                                subdict='inputs',
                                key=op_inputs[i],
                                value=inputs_memory[i],
                                overwrite=True
                            )

                else:
                    self.memory_context = self.updateDict(
                        self.memory_context,
                        subdict='weights',
                        key=op_weights,
                        value=weights_memory,
                        overwrite=False
                    )

                # output to main memory
                self.memory_context = self.updateDict(
                    self.memory_context,
                    subdict='outputs',
                    key=current_output,
                    value=output_memory,
                    overwrite=False
                )

                # output to main memory
                self.memory_context = self.updateDict(
                    self.memory_context,
                    subdict=None,
                    key='total_memory',
                    value=inst_total_memory,
                    overwrite=False,
                    add=True
                )

                if operator_idx > 0:
                    self.log_memory_view[-1] = self.memory_context

            else:
                op_inputs = _input_metadata[0]
                inputs_memory = _input_metadata[1]
                _all_valid_inputs = all([False if op_input is numpy.nan else True for op_input in op_inputs])

                if _all_valid_inputs:
                    # get inputs for operator from main memory
                    for i in range(len(op_inputs)):
                        if op_inputs[i] and inputs_memory[i]:
                            self.memory_context = self.updateDict(
                                self.memory_context,
                                subdict='inputs',
                                key=op_inputs[i],
                                value=inputs_memory[i],
                                overwrite=False
                            )

                else:
                    self.memory_context['inputs'] = {}

                if not _no_weights:
                    # get weights for operator from main memory
                    self.memory_context = self.updateDict(
                        self.memory_context,
                        subdict='weights',
                        key=op_weights,
                        value=weights_memory,
                        overwrite=False
                    )

                else:
                    self.memory_context['weights'] = {}

                # output to main memory
                self.memory_context = self.updateDict(
                    self.memory_context,
                    subdict='outputs',
                    key=current_output,
                    value=output_memory,
                    overwrite=False
                )

                # compute operators for operator
                self.memory_context = self.updateDict(
                    self.memory_context,
                    subdict=None,
                    key='compute',
                    value=compute_ops,
                    overwrite=False
                )

                # output to main memory
                self.memory_context = self.updateDict(
                    self.memory_context,
                    subdict=None,
                    key='total_memory',
                    value=inst_total_memory,
                    overwrite=False
                )

                self.log_memory_view.append(copy.deepcopy(self.memory_context))

        save_object = copy.deepcopy(self.log_memory_view)

        self.logData(self.log_files['memory_view'], self.log_memory_view)

        if plot_memory:
            # memory usage
            for entry in self.log_memory_view:
                self.log_memory_usage.append(entry['total_memory'])

            self.plotMemory(
                self.log_memory_usage,
                None,
                None,
                steps=None,
                display=True
            )

        return save_object


    def run_with_cache(
            self,
            local_memory_size: int,
            cache_size: int,
            final_outputs: tuple,
            plot_memory: bool = False
    ) -> None:
        # cache size (in MB)
        self.cache_size = cache_size

        log_memory_context = self.generate_view(
            memory_size=local_memory_size,
            plot_memory=False
        )

        self.reset()

        operators_sequence = log_memory_context   
        self.outputs_sequence = self.outputs_profile['Output Name']

        # operators are in sequence
        for operator_idx, memory_profile in enumerate(operators_sequence):
            self.main_memory_context = {'id': operator_idx + 1, 'inputs': {}, 'outputs': {}}

            # operator fusion ids
            operation_fusion_ids = memory_profile['id']

            if isinstance(operation_fusion_ids, int):
                operation_fusion_ids = (operation_fusion_ids,)

            else:
                operation_fusion_ids = tuple(operation_fusion_ids)

            # inputs for operator
            op_inputs = list(memory_profile['inputs'].keys())
            inputs_memory = list(memory_profile['inputs'].values())

            # weights for operator
            op_weights = list(memory_profile['weights'].keys())
            weights_memory = list(memory_profile['weights'].values())

            # if inputs are not in cache, they have to
            # be pulled from main memory
            for i, op_input in enumerate(op_inputs):
                # if output is in cache, retrieve it from cache
                if self.checkKeyinDict(self.cache_context[self.cache_parentKey], op_input):
                    _ = self.retrieveKeyfromCache(op_input)

                else:
                    # input should be pulled in from main memory
                    self.main_memory_context = self.updateDict(
                        self.main_memory_context,
                        subdict='inputs',
                        key=op_input,
                        value=inputs_memory[i],
                        overwrite=False
                    )

            # weights should be pulled in from main memory
            for w, op_weight in enumerate(op_weights):
                self.main_memory_context = self.updateDict(
                    self.main_memory_context,
                    subdict='weights',
                    key=op_weight,
                    value=weights_memory[w],
                    overwrite=False
                )

            # current operator generates output
            current_outputs = list(memory_profile['outputs'].keys())
            outputs_memory = list(memory_profile['outputs'].values())

            for output_idx, current_output in enumerate(current_outputs):
                output_memory = outputs_memory[output_idx]

                track_output_entry = self.outputs_profile[self.outputs_profile['Output Name'] == current_output]

                output_id = track_output_entry.index.item() + 1
                input_indices = track_output_entry['Input Node Index'].item()
                frequency = track_output_entry['Frequency'].item()

                if isinstance(input_indices, str):
                    input_indices = [int(elem) for elem in input_indices.split(' ')]
                    next_input_id = input_indices[0]
                    imm_cachability = next_input_id - output_id

                # if output is the final output,
                # output is pushed to main memory
                elif current_output in final_outputs:
                    self.main_memory_context = self.updateDict(
                        self.main_memory_context,
                        subdict='outputs',
                        key=current_output,
                        value=output_memory,
                        overwrite=False
                    )

                    continue

                else:
                    continue

                # check if output is an intermediate output
                # (consumed within the operator after fusion)
                intersection = set(operation_fusion_ids).intersection(set(input_indices))

                # is the output input to other operators?
                # if no, output can be discarded
                if (len(intersection) != len(input_indices)
                    and self.evaluateOutput(frequency, imm_cachability)):
                    # if output memory size > cache size, the output needs to be pushed to main memory
                    # if it is an input to other operators
                    if output_memory >= self.cache_size:
                        self.main_memory_context = self.updateDict(
                            self.main_memory_context,
                            subdict='outputs',
                            key=current_output,
                            value=output_memory,
                            overwrite=False
                        )

                    else:
                        # else check if the output can be cached and cache it
                        if not self.checkKeyinDict(self.cache_context[self.cache_parentKey], current_output):
                            _ = self.updateCache(
                                current_output,
                                (operator_idx + 1, output_id, next_input_id, frequency, imm_cachability, output_memory)
                            )

                            # update input_indices for later use
                            updated_input_indices = input_indices[1:]
                            updated_input_indices = (' '.join([str(elem) for elem in updated_input_indices])
                                                    if updated_input_indices else numpy.nan)

                            self.outputs_profile['Input Node Index'].at[output_idx] = updated_input_indices


            self.cache_context = self.updateDict(
                self.cache_context,
                subdict=None,
                key='cache_occupied',
                value=round(self.cache_occupied, self.rounding_decimal),
                overwrite=True
            )

            # log data
            self.log_cache_view.append(copy.deepcopy(self.cache_context))

            # read and write memory
            total_read_memory = (sum(list(self.main_memory_context['inputs'].values()))
                                 + sum(weights_memory))

            self.main_memory_context = self.updateDict(
                self.main_memory_context,
                subdict=None,
                key='total_read_memory',
                value=round(total_read_memory, self.rounding_decimal),
                overwrite=False
            )

            self.main_memory_context = self.updateDict(
                self.main_memory_context,
                subdict=None,
                key='total_write_memory',
                value=round(sum(list(self.main_memory_context['outputs'].values())),
                            self.rounding_decimal),
                overwrite=False
            )

            self.log_memory_view.append(copy.deepcopy(self.main_memory_context))

        self.logData(self.log_files['cache_view'], self.log_cache_view)
        self.logData(self.log_files['main_memory_context'], self.log_memory_view)

        if plot_memory:
            # memory usage
            for entry in self.log_cache_view:
                self.log_cache_memory_usage.append(entry['cache_occupied'])

            self.plotMemory(
                self.log_cache_memory_usage,
                None,
                None,
                steps=None,
                for_cache=True,
                display=True
            )

