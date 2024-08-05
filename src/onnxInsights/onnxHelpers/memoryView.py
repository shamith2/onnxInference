# Local Memory and Cache View

import copy
import os
from pathlib import Path
from typing import Any

import numpy
import pandas

import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__producer__ = "memoryView"
__version__ = "0.2.3"


# memory view
class memoryView:
    def __init__(
            self,
            model_dir: str,
            model_profile: str,
            outputs_profile: str,
            frequency_threshold: int = 1,
            imm_cachability_threshold: int = 1
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
                                                'main_memory_context.json'),
            'cache_main_memory_context': os.path.join(self.mem_view_logs_directory,
                                                      'cache_main_memory_context.json'),
        }

        # hyperparameters
        self.frequency_threshold = frequency_threshold
        self.imm_cachability_threshold = imm_cachability_threshold
        self.rounding_decimal = 6


    def reset(
            self
    ) -> None:
        self.log_cache_view = []
        self.log_memory_view = []
        self.log_main_memory_context = []

        self.model_profile = pandas.read_csv(os.path.join(self.profile_logs_directory, self.model_dataframe_file))
        self.outputs_profile = pandas.read_csv(os.path.join(self.profile_logs_directory, self.outputs_database_file))

        # drop last row since it contains totals
        self.model_profile.drop(self.model_profile.tail(1).index, inplace=True)

        self.cache_occupied = 0.0
        self.memory_occupied = 0.0

        self.cache_context = {'entries': {}}
        self.local_memory = {}
        self.main_memory_context = {}

        self.outputs_sequence = None
        self.output_index_seq = None
        self.output_freq_seq = None


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
        if ((frequency >= self.frequency_threshold) and 
            (imm_cachability >= self.imm_cachability_threshold)):
            return True

        else:
            return False


    def evictKeyfromCache(
            self,
            key: str,
            memory: float
    ) -> None:
        del self.cache_context['entries'][key]
        self.cache_occupied -= memory

        self.refreshCache()


    def refreshCache(
            self
    ) -> None:
        # sort output_priority dict by frequency and imm_cachability
        # output with high imm_cachability implies that the output
        # needs to wait longer before it's used but high frequency
        # implies the output is used frequently
        self.cache_context['entries'] = dict(sorted(self.cache_context['entries'].items(),
                                         key=lambda x: (x[1]['frequency'], -1 * x[1]['imm_cachability']),
                                         reverse=True))


    def logData(
            self,
            filename: str,
            log_data: list
    ) -> None:
        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=4, separators=(',',': '))


    def updateCache(
            self,
            key: str,
            input_indices: str,
            value: tuple[int, float]
    ) -> int:
        input_indices = [int(elem) for elem in input_indices.split(' ')]

        frequency = self.output_freq_seq[self.outputs_sequence == key].item()
        output_index = self.output_index_seq[self.outputs_sequence == key].item()
        imm_cachability = input_indices[0] - output_index
        
        output_idx, output_memory = value
        operator_idx = output_idx - 1

        # evaluate if output is worth storing in cache
        if self.evaluateOutput(frequency, imm_cachability):
            self.cache_occupied += output_memory
        
        # if output is not worth storing in cache
        # push it to main memory
        else:
            self.main_memory_context = self.updateDict(
                self.main_memory_context,
                subdict='outputs',
                key=key,
                value=output_memory,
                overwrite=False
            )

            return 1

        # if output can fit in cache, add it to cache
        if self.cache_occupied < self.cache_size:
            # add output to cache
            self.cache_context = self.updateDict(
                self.cache_context,
                subdict='entries',
                key=key,
                value={
                    'output_id': output_idx,
                    'frequency': frequency,
                    'imm_cachability': imm_cachability,
                    'memory': output_memory
                },
                overwrite=False
            )

            self.refreshCache()

            # update input_indices for later use
            updated_input_indices = input_indices[1:]
            updated_input_indices = (' '.join([str(elem) for elem in updated_input_indices])
                                    if updated_input_indices else numpy.nan)
            
            self.outputs_profile['Input Node Index'].at[operator_idx] = updated_input_indices

        # else evit least priority output from cache which according to the sorting
        # is the last entry of the cache context
        else:
            while self.cache_occupied < self.cache_size:
                dict_keys, dict_values = zip(*self.cache_context['entries'].items())
                self.evictKeyfromCache(dict_keys[-1], dict_values[-1]['memory'])

        return 0


    def retrieveKeyfromCache(
            self,
            key: str
    ) -> int:
        output_dict = self.cache_context['entries'][key]

        output_idx = output_dict['output_id']
        frequency = output_dict['frequency']
        imm_cachability = output_dict['imm_cachability']
        output_memory = output_dict['memory']

        operator_idx = output_idx - 1

        input_indices = self.outputs_profile['Input Node Index'].at[operator_idx]

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
        
        self.outputs_profile['Input Node Index'].at[operator_idx] = updated_input_indices

        frequency -= 1
        # output_index = self.output_index_seq[self.outputs_sequence == key].item()
        # assert output_index == output_idx
        imm_cachability = input_indices[0] - output_idx

        # evaluate if output is worth saving in the cache
        if self.evaluateOutput(frequency, imm_cachability):
            self.cache_context = self.updateDict(
                    self.cache_context,
                    subdict='entries',
                    key=key,
                    value={
                        'output_id': output_idx,
                        'frequency': frequency,
                        'imm_cachability': imm_cachability,
                        'memory': output_memory
                    },
                    overwrite=True
                )

            self.refreshCache()

        else:
            self.evictKeyfromCache(key, output_memory)

            self.main_memory_context = self.updateDict(
                self.main_memory_context,
                subdict='outputs',
                key=key,
                value=output_memory,
                overwrite=False
            )

            return 2

        return 0


    def run_without_cache(
            self,
            memory_size: int
    ) -> None:
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
            self.main_memory_context = {}
            _no_weights = False

            # current operator to be executed
            _ = operators_sequence[operator_idx]

            # compute operations of operator (in GOPS)
            compute_ops = round(float(compute_operations_sequence[operator_idx]) / 1e6,
                                self.rounding_decimal)

            self.main_memory_context = self.updateDict(
                self.main_memory_context,
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
                output_list = self.log_main_memory_context[-1]['outputs'].keys()
                _prev_memory_adder = self.log_main_memory_context[-1]['total_memory']

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
                    self.main_memory_context = self.log_main_memory_context[-1]

                self.main_memory_context = self.updateDict(
                    self.main_memory_context,
                    subdict=None,
                    key='id',
                    value=operator_idx + 1,
                    overwrite=False
                )

                # compute operators for operator
                self.main_memory_context = self.updateDict(
                    self.main_memory_context,
                    subdict=None,
                    key='compute',
                    value=compute_ops,
                    overwrite=False,
                    add=True
                )

                if _no_weights:
                    for i in range(len(op_inputs)):
                        if op_inputs[i] and inputs_memory[i]:
                            self.main_memory_context = self.updateDict(
                                self.main_memory_context,
                                subdict='inputs',
                                key=op_inputs[i],
                                value=inputs_memory[i],
                                overwrite=True
                            )

                else:
                    self.main_memory_context = self.updateDict(
                        self.main_memory_context,
                        subdict='weights',
                        key=op_weights,
                        value=weights_memory,
                        overwrite=False
                    )

                # output to main memory
                self.main_memory_context = self.updateDict(
                    self.main_memory_context,
                    subdict='outputs',
                    key=current_output,
                    value=output_memory,
                    overwrite=False
                )

                # output to main memory
                self.main_memory_context = self.updateDict(
                    self.main_memory_context,
                    subdict=None,
                    key='total_memory',
                    value=inst_total_memory,
                    overwrite=False,
                    add=True
                )

                if operator_idx > 0:
                    self.log_main_memory_context[-1] = self.main_memory_context

            else:
                op_inputs = _input_metadata[0]
                inputs_memory = _input_metadata[1]
                _all_valid_inputs = all([False if op_input is numpy.nan else True for op_input in op_inputs])

                if _all_valid_inputs:
                    # get inputs for operator from main memory
                    for i in range(len(op_inputs)):
                        if op_inputs[i] and inputs_memory[i]:
                            self.main_memory_context = self.updateDict(
                                self.main_memory_context,
                                subdict='inputs',
                                key=op_inputs[i],
                                value=inputs_memory[i],
                                overwrite=False
                            )

                else:
                    self.main_memory_context['inputs'] = {}

                if not _no_weights:
                    # get weights for operator from main memory
                    self.main_memory_context = self.updateDict(
                        self.main_memory_context,
                        subdict='weights',
                        key=op_weights,
                        value=weights_memory,
                        overwrite=False
                    )

                else:
                    self.main_memory_context['weights'] = {}

                # output to main memory
                self.main_memory_context = self.updateDict(
                    self.main_memory_context,
                    subdict='outputs',
                    key=current_output,
                    value=output_memory,
                    overwrite=False
                )

                # compute operators for operator
                self.main_memory_context = self.updateDict(
                    self.main_memory_context,
                    subdict=None,
                    key='compute',
                    value=compute_ops,
                    overwrite=False
                )

                # output to main memory
                self.main_memory_context = self.updateDict(
                    self.main_memory_context,
                    subdict=None,
                    key='total_memory',
                    value=inst_total_memory,
                    overwrite=False
                )

                self.log_main_memory_context.append(self.main_memory_context)

        self.logData(self.log_files['main_memory_context'], self.log_main_memory_context)


    def run_with_cache(
            self,
            cache_size: int
    ) -> None:
        # cache size (in MB)
        self.cache_size = cache_size

        self.reset()

        operators_sequence = self.model_profile['Node']

        inputs_sequence = self.model_profile['Inputs Name']
        inputs_memory_seq = self.model_profile['Inputs Memory (in MB)']
        
        self.outputs_sequence = self.outputs_profile['Output Name']
        outputs_memory_seq = self.outputs_profile['Memory (in MB)']

        self.output_index_seq = self.outputs_profile['Output Node Index']
        self.output_freq_seq = self.outputs_profile['Frequency']

        # assert sequence_length == len(outputs_sequence)
        sequence_length = len(operators_sequence)

        # operators are in sequence
        for operator_idx in range(sequence_length):
            self.main_memory_context = {}

            # current operator to be executed
            _ = operators_sequence[operator_idx]

            # inputs for operator
            op_inputs = inputs_sequence.at[operator_idx]
            inputs_memory = inputs_memory_seq.at[operator_idx]

            if ((not isinstance(op_inputs, str) or not isinstance(inputs_memory, str)) 
                or (operator_idx == sequence_length - 1)):
                continue

            # if inputs are not in cache, they have to
            # be pulled from main memory
            for op_input in op_inputs.split(' '):
                # if output is in cache, retrieve it from cache
                if self.checkKeyinDict(self.cache_context['entries'], op_input):
                    _ = self.retrieveKeyfromCache(op_input)

                else:
                    # input should be pulled in from main memory
                    pass

            # execute current operator
            _ = operators_sequence[operator_idx]

            self.main_memory_context = self.updateDict(
                self.main_memory_context,
                subdict=None,
                key='id',
                value=(operator_idx + 1),
                overwrite=False
            )

            # current operator generates output
            current_output = self.outputs_sequence[operator_idx]
            output_memory = outputs_memory_seq[operator_idx]

            # is the output input to other operators?
            input_indices = self.outputs_profile['Input Node Index'][operator_idx]

            # if no, output can be discarded
            if not isinstance(input_indices, str):
                continue

            # if output memory size > cache size, the output needs
            # to be pushed to main memory if it is an input to other
            # operators
            if (output_memory >= self.cache_size):
                self.main_memory_context = self.updateDict(
                    self.main_memory_context,
                    subdict='outputs',
                    key=current_output,
                    value=output_memory,
                    overwrite=False
                )

            else:
                # else check if the output can be cached and cache it
                if not self.checkKeyinDict(self.cache_context['entries'], current_output):
                    _ = self.updateCache(
                        current_output,
                        input_indices,
                        (operator_idx + 1, output_memory)
                    )

            self.cache_context = self.updateDict(
                self.cache_context,
                subdict=None,
                key='cache_occupied',
                value={
                        'remaining_memory': (self.cache_size - self.cache_occupied)
                    },
                overwrite=True
            )

            # log data
            self.log_cache_view.append(copy.deepcopy(self.cache_context))

            try:
                _ = self.main_memory_context['outputs']
                self.log_main_memory_context.append(self.main_memory_context)
            except KeyError:
                pass

        # the last output needs to go to main memory
        self.main_memory_context = self.updateDict(
            self.main_memory_context,
            subdict='outputs',
            key=self.outputs_sequence[sequence_length - 1],
            value=outputs_memory_seq[sequence_length - 1],
            overwrite=False
        )

        self.log_main_memory_context.append(self.main_memory_context)

        self.logData(self.log_files['cache_view'], self.log_cache_view)
        self.logData(self.log_files['cache_main_memory_context'], self.log_main_memory_context)

