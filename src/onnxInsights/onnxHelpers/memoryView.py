# Local Memory and Cache View

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
__version__ = "0.2.0"


# local memory view
class lmemoryView:
    def __init__(
            self,
            model_dir: str,
            model_profile: str,
            outputs_profile: str
    ):
        self.root = Path(__file__).parents[3].resolve()
        self.workspace = Path(__file__).parent.resolve()

        model_dir = '_'.join(model_dir.split(' ')).lower()

        self.prof_directory = os.path.join(self.root, 'results', 'onnxProfile')
        self.profile_logs_directory = os.path.join(self.prof_directory, 'logs', model_dir)
        self.cache_logs_directory = os.path.join(self.prof_directory, 'logs', model_dir, 'lmemory')

        for p in [self.prof_directory, self.cache_logs_directory]:
            if not os.path.exists(p):
                os.makedirs(p)

        self.log_files = [os.path.join(self.cache_logs_directory, 'local_memory_view.json'),
                          os.path.join(self.cache_logs_directory, 'from_main_memory.json'),
                          os.path.join(self.cache_logs_directory, 'to_main_memory.json')]

        self.log_cache_view = []
        self.log_from_main_memory = []
        self.log_to_main_memory = []

        self.model_profile = pandas.read_csv(os.path.join(self.profile_logs_directory, model_profile))
        self.outputs_profile = pandas.read_csv(os.path.join(self.profile_logs_directory, outputs_profile))

        # drop last row since it contains totals
        self.model_profile.drop(self.model_profile.tail(1).index, inplace=True)

        self.memory_occupied = 0.0

        self.local_memory = {}
        self.memory_from_main_memory = {}
        self.memory_to_main_memory = {}
    

    def run(
            self
    ) -> None:
        operators_sequence = self.model_profile['Node']

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
            # inputs for operator
            op_inputs = inputs_sequence.at[operator_idx]

            # weights for operator
            op_weights = weights_sequence.at[operator_idx]
            weights_memory = weights_memory_seq[operator_idx]

            # execute current operator
            _ = operators_sequence[operator_idx]

            # current operator generates output
            current_output = outputs_sequence[operator_idx]
            output_memory = outputs_memory_seq[operator_idx]

            print(op_weights, weights_memory)
            cc





# cache view
class cacheView:
    def __init__(
            self,
            model_dir: str,
            model_profile: str,
            outputs_profile: str,
            cache_size: int,
            frequency_threshold: int = 1,
            imm_cachability_threshold: int = 1
    ):
        self.root = Path(__file__).parents[3].resolve()
        self.workspace = Path(__file__).parent.resolve()

        model_dir = '_'.join(model_dir.split(' ')).lower()

        self.prof_directory = os.path.join(self.root, 'results', 'onnxProfile')
        self.profile_logs_directory = os.path.join(self.prof_directory, 'logs', model_dir)
        self.cache_logs_directory = os.path.join(self.prof_directory, 'logs', model_dir, 'cache')

        for p in [self.prof_directory, self.cache_logs_directory]:
            if not os.path.exists(p):
                os.makedirs(p)

        self.log_files = [os.path.join(self.cache_logs_directory, 'cache_view.json'),
                          os.path.join(self.cache_logs_directory, 'to_main_memory.json')]

        self.log_cache_view = []
        self.log_to_main_memory = []

        self.model_profile = pandas.read_csv(os.path.join(self.profile_logs_directory, model_profile))
        self.outputs_profile = pandas.read_csv(os.path.join(self.profile_logs_directory, outputs_profile))

        # drop last row since it contains totals
        self.model_profile.drop(self.model_profile.tail(1).index, inplace=True)
        
        # hyperparameters
        self.cache_size = cache_size # cache size (in MB)
        self.frequency_threshold = frequency_threshold
        self.imm_cachability_threshold = imm_cachability_threshold

        self.cache_occupied = 0.0

        self.output_priority = {}
        self.memory_to_main_memory = {}

        self.outputs_sequence = None
        self.output_index_seq = None
        self.output_freq_seq = None


    def updateDict(
            self,
            dictionary: dict,
            key: str,
            value: Any,
            overwrite: bool = True
    ) -> dict:
        if dictionary.get(key, None) is None or overwrite:
            dictionary[key] = (value,) if not isinstance(value, tuple) else value
        
        else:
            dictionary[key] += (value,)

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
        del self.output_priority[key]
        self.cache_occupied -= memory

        self.refreshCache()


    def refreshCache(
            self
    ) -> None:
        # sort output_priority dict by imm_cachability and output memory
        # output with high imm_cachability implies that the output
        # needs to wait longer before it's used
        self.output_priority = dict(sorted(self.output_priority.items(),
                                           key=lambda x: (x[1][1], -1 * x[1][2]),
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

        frequency = self.output_index_seq[self.outputs_sequence == key].item()
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
            self.memory_to_main_memory = self.updateDict(
                self.memory_to_main_memory,
                key,
                output_memory,
                overwrite=None
            )

            return 1

        # if output can fit in cache, add it to cache
        if self.cache_occupied < self.cache_size:
            # add output to cache
            self.output_priority = self.updateDict(
                self.output_priority,
                key,
                (output_idx, imm_cachability, output_memory),
                overwrite=None
            )

            self.refreshCache()

            # update input_indices for later use
            updated_input_indices = input_indices[1:]
            updated_input_indices = (' '.join([str(elem) for elem in updated_input_indices])
                                    if updated_input_indices else numpy.nan)
            
            self.outputs_profile['Input Node Index'].at[operator_idx] = updated_input_indices

        # else evit least priority output from cache
        else:
            while self.cache_occupied < self.cache_size:
                dict_keys, dict_values = zip(*self.output_priority.items())
                self.evictKeyfromCache(dict_keys[0], dict_values[0][2])

        return 0


    def retrieveKeyfromCache(
            self,
            key: str
    ) -> int:
        output_idx, imm_cachability, output_memory = self.output_priority[key]
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

        frequency = self.output_index_seq[self.outputs_sequence == key].item()
        # output_index = self.output_index_seq[self.outputs_sequence == key].item()
        # assert output_index == output_idx
        imm_cachability = input_indices[0] - output_idx

        # evaluate if output is worth saving in the cache
        if self.evaluateOutput(frequency, imm_cachability):
            self.output_priority = self.updateDict(
                    self.output_priority,
                    key,
                    (output_idx, imm_cachability, output_memory),
                    overwrite=True
                )

            self.refreshCache()

        else:
            self.evictKeyfromCache(key, output_memory)

            self.memory_to_main_memory = self.updateDict(
                self.memory_to_main_memory,
                key,
                output_memory,
                overwrite=None
            )

            return 2

        return 0


    def run(
            self
    ) -> None:
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
            # inputs for operator
            op_inputs = inputs_sequence.at[operator_idx]
            
            if ((not isinstance(op_inputs, str) and not inputs_memory_seq[operator_idx]) 
                or (operator_idx == sequence_length - 1)):
                continue

            # if inputs are not in cache, they have to
            # be pulled from main memory
            for op_input in op_inputs.split(' '):
                # if output is in cache, retrieve it from cache
                if self.checkKeyinDict(self.output_priority, op_input):
                    _ = self.retrieveKeyfromCache(op_input)

                else:
                    # input should be pulled in from main memory
                    pass

            # execute current operator
            _ = operators_sequence[operator_idx]

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
                self.memory_to_main_memory = self.updateDict(
                    self.memory_to_main_memory,
                    current_output,
                    output_memory,
                    overwrite=None
                )

            else:
                # else check if the output can be cached and cache it
                if not self.checkKeyinDict(self.output_priority, current_output):
                    _ = self.updateCache(
                        current_output,
                        input_indices,
                        (operator_idx + 1, output_memory)
                    )

            # log data
            self.log_cache_view.append(self.output_priority)
            self.log_to_main_memory.append(self.memory_to_main_memory)


        # the last output needs to go to main memory
        self.memory_to_main_memory = self.updateDict(
            self.memory_to_main_memory,
            self.outputs_sequence[sequence_length - 1],
            outputs_memory_seq[sequence_length - 1],
            overwrite=None
        )

        self.log_to_main_memory.append(self.memory_to_main_memory)

        self.logData(self.log_files[0], self.log_cache_view)
        self.logData(self.log_files[1], self.log_to_main_memory)

