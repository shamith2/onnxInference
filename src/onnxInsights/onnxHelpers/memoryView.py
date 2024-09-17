# Local Memory and Cache View

import copy
import os
from pathlib import Path
from typing import Any, Union, Optional

import numpy
import pandas
from matplotlib import pyplot as plt

import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__producer__ = "memoryView"
__version__ = "0.2.5"


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
        self.plots_directory = os.path.join(self.prof_directory, 'logs', model_dir,
                                            'memoryView', 'plots')

        for p in [self.prof_directory, self.mem_view_logs_directory, self.plots_directory]:
            if not os.path.exists(p):
                os.makedirs(p)

        self.logfile_extension = '.json'

        self.log_files = {
            'memory_view': os.path.join(self.mem_view_logs_directory, 'local_memory_view' + self.logfile_extension),
            'cache_view': os.path.join(self.mem_view_logs_directory, 'cache_view' + self.logfile_extension),
            'main_memory_context': os.path.join(self.mem_view_logs_directory,
                                                'main_memory_context' + self.logfile_extension)
        }

        # hyperparameters
        self.frequency_threshold = 1
        self.imm_cachability_threshold = 1
        self.rounding_decimal = 6


    def reset(
            self
    ) -> None:
        self.model_profile = pandas.read_csv(os.path.join(self.profile_logs_directory, self.model_dataframe_file))
        self.outputs_profile = pandas.read_csv(os.path.join(self.profile_logs_directory, self.outputs_database_file))

        # drop last row since it contains totals
        self.model_profile.drop(self.model_profile.tail(1).index, inplace=True)

        self.cache_occupied = 0.0
        self.memory_occupied = 0.0
        self.local_memory_used = 0.0

        self.log_memory_view = []
        self.log_cache_view = []
        self.store_input_id = {}

        self.cache_parentKey = 'entries'
        self.cache_context = {self.cache_parentKey: {}, 'local_memory': {}}

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
        """
        Scenarios:
        +-----------+-----------------+--------------+
        | frequency | imm_cachability | result       |
        +-----------+-----------------+--------------+
        | 1         | 1               | local_memory |
        +-----------+-----------------+--------------+
        | > 1       | 1               | local_memory |
        +-----------+-----------------+--------------+
        | 1         | > 1             | cache        |
        +-----------+-----------------+--------------+
        | > 1       | > 1             | cache        |
        +-----------+-----------------+--------------+
        """

        if self.frequency_threshold > 1 or self.imm_cachability_threshold > 1:
            logging.warning('frequency threshold or imm_cachablity threshold is greater than 1. check if this is intended')

        if imm_cachability <= self.imm_cachability_threshold:
            return 'local_memory'

        else:
            return 'cache'


    def checkOperatorFusion(
            self,
            input_indices: tuple,
            operator_fusion_ids: tuple
    ) -> bool:
        # check if output is an intermediate output
        # (consumed within the operator after fusion)
        intersection = set(operator_fusion_ids).intersection(set(input_indices))

        if len(intersection) == len(input_indices):
            return True

        else:
            return False


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
            save_path: str,
            first_variable: list,
            second_variable: list,
            attributes: tuple[dict],
            min_val: float,
            max_val: float,
            steps: int,
            fig_size: tuple[int],
            for_cache: bool = False
    ) -> None:
        if not for_cache and second_variable:
            logging.warning("second_variable should be None if not for_cache. Setting second_variable to None")
            second_variable = None

            if not min_val:
                min_val = min(first_variable)

            if not max_val:
                max_val = max(first_variable)

            filtered_mem_usage = self.findOutliers(
                first_variable,
                min_val,
                max_val
            )

        else:
            if second_variable:
                assert len(first_variable) == len(second_variable), "first_variable and second variable should have same length"
                assert len(attributes) == 2, "attributes have to provided and should match the order [first_variable second_variable]"

            filtered_mem_usage = first_variable

        num_plots = 2 if second_variable else 1

        fig, ax = plt.subplots(num_plots, figsize=fig_size)

        first_fig = ax[0] if second_variable else ax
        axes_list = [first_fig, ax[1]] if second_variable else [first_fig]

        x = range(len(filtered_mem_usage))

        if second_variable:
            min_val = [min(filtered_mem_usage), min(second_variable)]
            max_val = [max(filtered_mem_usage), max(second_variable)]

        else:
            min_val = [min(filtered_mem_usage)]
            max_val = [max(filtered_mem_usage)]

        _ = first_fig.scatter(
            x,
            filtered_mem_usage,
            c='g',
            label=attributes[0]['legend'],
        )

        # set axes labels and title
        first_fig.set_xticks(x)
        y_range = (numpy.linspace(min_val[0], max_val[0], steps, endpoint=True) if steps
                   else numpy.linspace(min_val[0], max_val[0], endpoint=True))
        first_fig.set_yticks(y_range)

        if second_variable:
            _ = ax[1].scatter(
                x,
                second_variable,
                c='y',
                label=attributes[1]['legend']
            )

            # set axes labels and title
            ax[1].set_xticks(x)
            y_range = (numpy.linspace(min_val[1], max_val[1], steps, endpoint=True) if steps
                       else numpy.linspace(min_val[1], max_val[1], endpoint=True))
            ax[1].set_yticks(y_range)


        for i, axis in enumerate(axes_list):
            axis.set_xlabel(attributes[i]['xlabel'])      
            axis.set_ylabel(attributes[i]['ylabel'])

            axis.set_title(attributes[i]['figtitle'])

            axis.legend(loc='best')
            axis.tick_params(bottom=True, labelbottom=False)

        fig.suptitle(attributes[0]['suptitle'], fontweight='bold')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path)

        plt.close(fig)


    def evictKeyfromCache(
            self,
            key: str,
            memory: float
    ) -> None:
        del self.cache_context[self.cache_parentKey][key]
        self.cache_occupied -= memory

        self.refreshCache()


    def evictKeyfromLocalMemory(
            self,
            key: str
    ) -> None:
        memory = self.cache_context['local_memory'][key]
        del self.cache_context['local_memory'][key]
        self.local_memory_used -= memory


    def checkandFreeLocalMemory(
            self,
            key: str
    ) -> int:
        row_index = self.outputs_profile.loc[self.outputs_profile['Output Name'] == key].index.item()

        input_indices = self.outputs_profile['Fused Input Node Index'].at[row_index]
        entry_idx = self.outputs_profile['Fused Output Node Index'].at[row_index]

        # is output does not have valid input_indices => frequency = 0,
        # the output can be discarded
        if not isinstance(input_indices, str):
            self.evictKeyfromLocalMemory(key)
            return 1

        input_indices = [int(elem) for elem in input_indices.split(' ')]

        frequency = len(input_indices)
        memory = self.cache_context['local_memory'][key]

        # scenario of next_input_id is not reached because
        # it's detected earlier through input_indices
        next_input_id = self.updateInputIndices(key, input_indices)

        if self.store_input_id.get(row_index, None) is None:
            imm_cachability = next_input_id - entry_idx

        else:
            imm_cachability = next_input_id - self.store_input_id[row_index]

        # save input_id for later use
        self.store_input_id[row_index] = next_input_id

        # key no longer needed in local memory but is needed
        # as input later
        if self.evaluateOutput(frequency, imm_cachability) == 'cache':
            if memory >= self.cache_size:
                self.main_memory_context = self.updateDict(
                    self.main_memory_context,
                    subdict='outputs',
                    key=key,
                    value=memory,
                    overwrite=True
                )

                self.evictKeyfromLocalMemory(key)
                return 2

            else:
                _ = self.updateCache(
                    key=key,
                    value=(-1, next_input_id, frequency, imm_cachability, memory)
                )

                self.evictKeyfromLocalMemory(key)
                return 3

        return 0


    def refreshCache(
            self
    ) -> None:
        """
        sort output_priority dict by frequency, imm_cachability, next_input_id and memory
        sorted in reverse by the order of the metrics listed above:
            higher frequency, lower imm_cachability, lower next_input_id and higher memory is preferred
        
        output with high imm_cachability implies that the output
        needs to wait longer before it's used and high frequency
        implies the output is used frequently
        """
        self.cache_context[self.cache_parentKey] = dict(sorted(self.cache_context[self.cache_parentKey].items(),
                                                               key=lambda x: (x[1]['frequency'], -1 * x[1]['imm_cachability'], -1 * x[1]['next_input_id'], x[1]['memory']),
                                                               reverse=True))


    def updateCache(
            self,
            key: str,
            value: tuple[int, int, int, int, float]
    ) -> int:
        operator_id, next_input_id, frequency, imm_cachability, output_memory = value

        self.cache_occupied += output_memory

        # add output to cache
        self.cache_context = self.updateDict(
            self.cache_context,
            subdict=self.cache_parentKey,
            key=key,
            value={
                'operator_id': operator_id,
                'next_input_id': next_input_id,
                'frequency': frequency,
                'imm_cachability': imm_cachability,
                'memory': output_memory
            },
            overwrite=True
        )

        self.refreshCache()

        # if output cannot fit in cache, evit least priority output
        # from cache which according to the sorting,
        # which is the last entry of the cache context
        while self.cache_occupied > self.cache_size:
            dict_keys, dict_values = zip(*self.cache_context[self.cache_parentKey].items())

            evicted_key, evicted_memory = dict_keys[-1], dict_values[-1]['memory']

            self.evictKeyfromCache(evicted_key, evicted_memory)

            self.main_memory_context = self.updateDict(
                self.main_memory_context,
                subdict='outputs',
                key=evicted_key,
                value=evicted_memory,
                overwrite=True
            )

        return 0


    def pushtoLocalMemory(
            self,
            key: str,
            memory: float
    ) -> None:
        self.cache_context = self.updateDict(
            self.cache_context,
            subdict='local_memory',
            key=key,
            value=memory,
            overwrite=True
        )

        self.local_memory_used += memory


    def updateInputIndices(
            self,
            key: str,
            input_indices: list = None
    ) -> Union[str, int]:
        row_index = self.outputs_profile.loc[self.outputs_profile['Output Name'] == key].index.item()

        if not input_indices:
            input_indices = self.outputs_profile['Fused Input Node Index'].at[row_index]

            if isinstance(input_indices, str):
                input_indices = [int(elem) for elem in input_indices.split(' ')]

            else:
                return 'discard'

        # update input_indices for later use
        next_input_id = input_indices[0]
        updated_input_indices = input_indices[1:]

        if updated_input_indices:
            updated_input_indices = ' '.join([str(elem) for elem in updated_input_indices])

            self.outputs_profile['Fused Input Node Index'].at[row_index] = updated_input_indices

            return next_input_id

        else:
            self.outputs_profile['Fused Input Node Index'].at[row_index] = numpy.nan

            return next_input_id


    def retrieveKeyfromCache(
            self,
            key: str
    ) -> int:
        output_dict = self.cache_context[self.cache_parentKey][key]

        frequency = output_dict['frequency']
        output_memory = output_dict['memory']

        # if the key is no longer used as input to other operators,
        # discard it
        next_input_id = self.updateInputIndices(key, None)

        if next_input_id == 'discard':
            self.evictKeyfromCache(key, output_memory)
            return 1

        frequency -= 1
        imm_cachability = next_input_id - output_dict['next_input_id']

        # evaluate if output is worth saving in the local memory
        if (self.evaluateOutput(frequency, imm_cachability) == 'local_memory'
            and self.local_memory_used + output_memory < self.local_memory_size):
            self.evictKeyfromCache(key, output_memory)
            self.pushtoLocalMemory(key, output_memory)

        # if key is large for local memory,
        # keep it in cache until it's no longer required
        self.cache_context = self.updateDict(
            self.cache_context,
            subdict=self.cache_parentKey,
            key=key,
            value={
                'operator_id': output_dict['operator_id'],
                'next_input_id': next_input_id,
                'frequency': frequency,
                'imm_cachability': imm_cachability,
                'memory': output_memory
            },
            overwrite=True
        )

        self.refreshCache()

        return 0


    def recalculateInputIndices(
            self,
            log_memory_context: list[dict]
    ) -> None:
        # recalculate only if the dataframe does not have these columns
        try:
            self.outputs_profile.insert(2, 'Fused Output Node Index', pandas.Series())
            self.outputs_profile.insert(4, 'Fused Input Node Index', pandas.Series())
            self.outputs_profile.insert(6, 'Fused Frequency', pandas.Series())
            self.outputs_profile.insert(7, 'Fused Cachability', pandas.Series())
        except ValueError:
            return

        fused_operators_sequence = log_memory_context
        sequence_length = len(fused_operators_sequence)

        fused_op_index = {}

        # map original operator index to fused operator index
        # operator index are sorted
        for iter_idx in range(sequence_length):
            current_op_idx = fused_operators_sequence[iter_idx]['id']
            fused_op_idx = iter_idx + 1
            
            if isinstance(current_op_idx, int):
                fused_op_index[current_op_idx] = fused_op_idx

            else:
                for op_idx in current_op_idx:
                    fused_op_index[op_idx] = fused_op_idx

        # operators are in sequence
        for iter_idx in range(sequence_length):
            current_memory_profile = fused_operators_sequence[iter_idx]
            current_outputs = current_memory_profile['outputs']

            for current_output in current_outputs:
                # find output name
                row_index = self.outputs_profile.loc[self.outputs_profile['Output Name'] == current_output].index.item()

                # update output node index
                fused_output_id = fused_op_index[int(self.outputs_profile['Output Node Index'].at[row_index])]
                self.outputs_profile['Fused Output Node Index'].at[row_index] = fused_output_id

                # update input node index
                input_indices = self.outputs_profile['Input Node Index'].at[row_index]

                if isinstance(input_indices, str):
                    input_indices = [int(elem) for elem in input_indices.split(' ')]
                    
                    fused_input_indices = sorted(set([fused_op_index[input_id] for input_id in input_indices]) - {fused_output_id})

                    updated_indices = ' '.join([str(elem) for elem in fused_input_indices]) if fused_input_indices else numpy.nan

                else:
                    updated_indices = numpy.nan

                self.outputs_profile['Fused Input Node Index'].at[row_index] = updated_indices

                # update cachability
                if updated_indices is not numpy.nan:
                    self.outputs_profile['Fused Cachability'].at[row_index] = max(fused_input_indices) - fused_output_id
                    self.outputs_profile['Fused Frequency'].at[row_index] = len(fused_input_indices)
                
                else:
                    self.outputs_profile['Fused Cachability'].at[row_index] = 0
                    self.outputs_profile['Fused Frequency'].at[row_index] = 0

        # save dataframe
        self.outputs_profile.to_csv(os.path.join(self.profile_logs_directory, self.outputs_database_file),
                                    index=False, mode='w')


    def generate_view(
            self,
            memory_size: int,
            save_path: Optional[str] = None,
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
            self.log_memory_usage = []

            # memory usage
            for entry in self.log_memory_view:
                self.log_memory_usage.append(entry['total_memory'])

            attributes = {
                'xlabel': 'Operator',
                'ylabel': 'Total Memory Size [in MB]',
                'legend': 'total memory',
                'suptitle': 'Memory Profile',
                'figtitle': '\nTotal Memory = Inputs Memory + Weights Memory + Outputs Memory'
            }

            self.plotMemory(
                save_path=save_path,
                first_variable=self.log_memory_usage,
                second_variable=None,
                attributes=(attributes,),
                min_val=None,
                max_val=None,
                steps=None,
                fig_size=(15, 12),
                for_cache=False
            )

        return save_object


    def run_with_cache(
            self,
            local_memory_size: int,
            cache_size: int,
            final_outputs: tuple,
            plot_memory: bool = False
    ) -> None:
        """
        Score Scenario:
            1. Inputs <- Memory: score = 0.0
            2. Weights <- Memory: score = 0.0
            3. Evict from Cache: score = -10.0
        """
        # memory and cache size (in MB)
        self.local_memory_size = local_memory_size
        self.cache_size = cache_size

        self.score = 0.0

        log_memory_context = self.generate_view(
            memory_size=local_memory_size,
            save_path=os.path.join(self.plots_directory,
                                   'memory_view_for_lmsz{}mb.png'.format(local_memory_size)),
            plot_memory=True,
        )

        self.recalculateInputIndices(log_memory_context)

        self.reset()

        operators_sequence = log_memory_context   
        self.outputs_sequence = self.outputs_profile['Output Name']

        # operators are in sequence
        for operator_idx, memory_profile in enumerate(operators_sequence):
            self.main_memory_context = {'id': operator_idx + 1, 'inputs': {}, 'outputs': {}}

            self.cache_context = self.updateDict(
                self.cache_context,
                subdict=None,
                key='id',
                value=operator_idx + 1,
                overwrite=True
            )

            # inputs for operator
            op_inputs = list(memory_profile['inputs'].keys())
            inputs_memory = list(memory_profile['inputs'].values())

            # weights for operator
            op_weights = list(memory_profile['weights'].keys())
            weights_memory = list(memory_profile['weights'].values())

            # if inputs are not in cache, they have to
            # be pulled from main memory
            for i, op_input in enumerate(op_inputs):
                # if input is in local memory, use it from local memory
                if self.checkKeyinDict(self.cache_context['local_memory'], op_input):
                    pass

                # if output is in cache, retrieve it from cache
                elif self.checkKeyinDict(self.cache_context[self.cache_parentKey], op_input):
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

            # check if local memory needs to be freed
            for key in list(self.cache_context['local_memory'].keys()):
                retval = self.checkandFreeLocalMemory(key)

                if retval == 2:
                    self.score -= 5.0

                elif retval == 3:
                    self.score -= 2.0

            # current operator generates output
            current_outputs = list(memory_profile['outputs'].keys())
            outputs_memory = list(memory_profile['outputs'].values())

            for output_idx, current_output in enumerate(current_outputs):
                output_memory = outputs_memory[output_idx]

                track_output_entry = self.outputs_profile[self.outputs_profile['Output Name'] == current_output]

                output_id = track_output_entry['Fused Output Node Index'].item()
                input_indices = track_output_entry['Fused Input Node Index'].item()
                frequency = track_output_entry['Fused Frequency'].item()

                if isinstance(input_indices, str):
                    input_indices = [int(elem) for elem in input_indices.split(' ')]
                    imm_cachability = input_indices[0] - output_id

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

                # is the output input an intermediate output?
                # if yes, output can be discarded
                else:
                    continue


                # check if current output can be stored in local memory
                if (self.evaluateOutput(frequency, imm_cachability) == 'local_memory'
                    and self.local_memory_used + output_memory < self.local_memory_size):
                        self.pushtoLocalMemory(current_output, output_memory)

                # or stored in cache
                else:
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

                        self.score -= 5.0

                    else:
                        # else cache the output
                        _ = self.updateCache(
                            current_output,
                            (operator_idx + 1, output_id, frequency, imm_cachability, output_memory)
                        )

                        _ = self.updateInputIndices(current_output, input_indices)

                        self.score -= 2.0


            self.cache_context = self.updateDict(
                self.cache_context,
                subdict=None,
                key='local_memory_occupied',
                value=round(self.local_memory_used, self.rounding_decimal),
                overwrite=True
            )

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
            self.log_cache_memory_usage = []
            self.log_local_memory_usage = []

            self.log_reads = []
            self.log_writes = []

            # memory usage
            for entry in self.log_cache_view:
                self.log_cache_memory_usage.append(entry['cache_occupied'])
                self.log_local_memory_usage.append(entry['local_memory_occupied'])
            
            for entry in self.log_memory_view:
                self.log_reads.append(entry['total_read_memory'])
                self.log_writes.append(entry['total_write_memory'])

            attributes_1 = {
                'xlabel': 'Timestep',
                'ylabel': 'Memory used [in MB]',
                'legend': 'cache occupied',
                'suptitle': 'Local Output Memory and Cache Profile',
                'figtitle': '\nCache Memory = Outputs Memory'
            }

            attributes_2 = {
                'xlabel': 'Timestep',
                'ylabel': 'Memory used [in MB]',
                'legend': 'local memory\noccupied',
                'figtitle': '\nLocal Output Memory = Outputs Memory'
            }

            attributes_3 = {
                'xlabel': 'Timestep',
                'ylabel': 'Memory [in MB]',
                'legend': 'read memory',
                'suptitle': 'Operator Reads and Writes to Main Memory',
                'figtitle': '\nRead Memory = Inputs Memory + Weights Memory'
            }

            attributes_4 = {
                'xlabel': 'Timestep',
                'ylabel': 'Memory [in MB]',
                'legend': 'write memory',
                'figtitle': '\nWrite Memory = Outputs Memory'
            }

            self.plotMemory(
                save_path=os.path.join(self.plots_directory,
                                       'cache_and_local_memory_usage_for_csz{}mb_lmsz{}mb.png'.format(cache_size, local_memory_size)),
                first_variable=self.log_cache_memory_usage,
                second_variable=self.log_local_memory_usage,
                attributes=(attributes_1, attributes_2),
                min_val=None,
                max_val=None,
                steps=20,
                fig_size=(15, 12),
                for_cache=True
            )

            self.plotMemory(
                save_path=os.path.join(self.plots_directory,
                                       'reads_and_writes_to_main_memory_for_csz{}mb_lmsz{}mb.png'.format(cache_size, local_memory_size)),
                first_variable=self.log_reads,
                second_variable=self.log_writes,
                attributes=(attributes_3, attributes_4),
                min_val=None,
                max_val=None,
                steps=20,
                fig_size=(15, 12),
                for_cache=True
            )

        return self.score

