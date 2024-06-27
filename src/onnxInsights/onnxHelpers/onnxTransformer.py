# This script contains functions for profiling and modifying ONNX graphs

import copy
import os

import numpy
import onnx
from onnx.helper import tensor_dtype_to_string

from typing import Any
from pathlib import Path

import shutil

import onnxruntime
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

from .onnxBenchmark import get_random_input

import pandas

import logging

logging.basicConfig(level=logging.INFO)
# logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

__producer__ = "onnxTransformer"
__version__ = "0.2.1"


DTYPES = {
    'TensorProto.BOOL': 1,
    'TensorProto.UINT8': 1,
    'TensorProto.INT8': 1,
    'TensorProto.UINT16': 2,
    'TensorProto.INT16': 2,
    'TensorProto.FLOAT16': 2,
    'TensorProto.INT32': 4,
    'TensorProto.FLOAT': 4,
    'TensorProto.INT64': 8,
    'TensorProto.DOUBLE': 8,
}


# Helper Functions

def checkandSaveModel(
    model: onnx.ModelProto,
    extension: str,
    save_directory: str,
    filename: str
) -> int:
    if filename.endswith(extension):
        filename_with_extension = os.path.join(save_directory, filename)
        filename = filename.removesuffix(extension)
    
    else:
        filename_with_extension = os.path.join(save_directory, filename + extension)

    for f in [filename_with_extension, filename + '.onnx_data']:
        path = os.path.join(save_directory, f)
        
        if os.path.exists(path):
            logging.warning('{} already exists. Removing exisiting file'.format(os.path.split(path)[-1]))
            os.remove(path)

    try:
        onnx.checker.check_model(model, check_custom_domain=True)
        
        onnx.save(model, filename_with_extension)

        logger.info("Model saved as {}".format(filename_with_extension))
    
    except ValueError:
        external_data = filename + '.onnx_data'

        onnx.save(model, filename_with_extension, save_as_external_data=True, all_tensors_to_one_file=True, location=external_data, size_threshold=1024)

        try:
            onnx.checker.check_model(filename_with_extension, check_custom_domain=True)

        except onnx.onnx_cpp2py_export.checker.ValidationError:
            # alternative for onnx.checker.check_model with custom domain operators
            _ = onnxruntime.InferenceSession(
                filename_with_extension,
                providers=["CPUExecutionProvider"]
            )

        logger.info("Model saved as {} with external data saved as {} in the same directory as the model".format(filename_with_extension, external_data))
    
    return 0


def _convert_shape_tuple_to_string(
        tuple_of_tuple: tuple[tuple[Any]],
        add: bool = False
) -> str:
    output = ''

    for i, shape_tuple in enumerate(tuple_of_tuple):
        output += 'x'.join(str(dim) for dim in shape_tuple)

        if i < len(tuple_of_tuple) - 1:
            output += ', '
    
    param_sum = 0

    if add:
        for out in output.split(', '):
            param_sum += int(out)
    
    return param_sum if add else output


class ONNXTransformer:
    def __init__(
            self
    ):
        self.extension = '.onnx'

        self.root = Path(__file__).parents[3].resolve()
        self.workspace = Path(__file__).parent.resolve()

        self.prof_directory = os.path.join(self.root, 'results', 'onnxProfile')
        self.infer_model_directory = os.path.join(self.prof_directory, 'models')
        self.profile_logs_directory = os.path.join(self.prof_directory, 'logs')

        for p in [self.prof_directory, self.infer_model_directory, self.profile_logs_directory]:
            if not os.path.exists(p):
                os.makedirs(p)

        self.node_input_dict = {}
        self.node_output_dict = {}
        self.node_wb_dict = {}


    # adapted from https://github.com/onnx/onnx/blob/main/onnx/tools/update_model_dims.py
    def _update_dim(
            self,
            tensor: onnx.ValueInfoProto,
            dim: Any,
            j: int,
            name: str
    ) -> None:
        dim_param_set: set[str] = set()

        def __init_dim_param_set(
            dim_param_set: set[str], value_infos: list[onnx.ValueInfoProto]
        ) -> None:
            for info in value_infos:
                shape = info.type.tensor_type.shape
                for dim in shape.dim:
                    if dim.HasField("dim_param"):
                        dim_param_set.add(dim.dim_param)

        __init_dim_param_set(dim_param_set, self.onnx_model.graph.input)  # type: ignore
        __init_dim_param_set(dim_param_set, self.onnx_model.graph.output)  # type: ignore
        __init_dim_param_set(dim_param_set, self.onnx_model.graph.value_info)  # type: ignore

        dim_proto = tensor.type.tensor_type.shape.dim[j]
        
        if isinstance(dim, int):
            if dim >= 0:
                if dim_proto.HasField("dim_value") and dim_proto.dim_value != dim:
                    raise ValueError(
                        "Unable to set dimension value to {} for axis {} of {}. Contradicts existing dimension value {}".format(
                            dim, j, name, dim_proto.dim_value
                        )
                    )
                
                dim_proto.dim_value = dim
            
            else:
                generated_dim_param = name + "_" + str(j)
                
                if generated_dim_param in dim_param_set:
                    raise ValueError(
                        "Unable to generate unique dim_param for axis {} of {}. Please manually provide a dim_param value".format(
                            j, name
                        )
                    )
                
                dim_proto.dim_param = generated_dim_param
        
        elif isinstance(dim, str):
            dim_proto.dim_param = dim
        
        else:
            raise ValueError(
                "Only int or str is accepted as dimension value, incorrect type: {}".format(type(dim))
            )
    

    def _verify_inputs_and_outputs(
        self,
        onnx_model_path: str,
        static_input_dims: list,
        static_output_dims: list
    ) -> str:
        self.onnx_model = onnx.load(onnx_model_path)

        for i, model_input in enumerate(self.onnx_model.graph.input):
            input_name = model_input.name
            
            for j, dim in enumerate(static_input_dims[i]):
                self._update_dim(model_input, dim, j, input_name)

        for i, model_output in enumerate(self.onnx_model.graph.output):
            output_name = model_output.name
            
            for j, dim in enumerate(static_output_dims[i]):
                self._update_dim(model_output, dim, j, output_name)

        intermediate_onnx_file = self.model_name + '_intermediate'
        
        save_path = os.path.join(self.infer_model_directory, intermediate_onnx_file + self.extension)

        assert checkandSaveModel(self.onnx_model, self.extension, self.infer_model_directory, intermediate_onnx_file) == 0, "checkandSaveModel() failed"

        return save_path


    def render(
        self,
        file_directory: str,
        filename: str
    ) -> None:
        return "Import-Csv " + os.path.join(file_directory, filename) + " | Out-GridView"


    def shapeInfer(
            self,
            model_name: str,
            onnx_model_file: str,
            static_input_dims: list,
            static_output_dims: list
    ) -> str:
        self.model_name = model_name

        intermediate_model_file = self._verify_inputs_and_outputs(onnx_model_file, static_input_dims, static_output_dims)

        logger.info("Performing symbolic shape inference")
        
        inferred_model = SymbolicShapeInference.infer_shapes(
            onnx.load(intermediate_model_file),
            int_max=2**31 - 1,
            auto_merge=False,
            guess_output_rank=False,
            verbose=0
        )

        inferred_model_file = os.path.join(self.infer_model_directory, self.model_name + '_inferred' + self.extension)

        assert checkandSaveModel(inferred_model, self.extension, self.infer_model_directory, self.model_name + '_inferred') == 0, "checkandSaveModel() failed"

        os.remove(intermediate_model_file)

        try:
            os.remove(intermediate_model_file.removesuffix(self.extension) + '.onnx_data')
        
        except FileNotFoundError:
            pass

        return inferred_model_file


    def updateTensorandWeightDict(
            self,
            model_input_list: list,
            model_output_list: list,
            model_value_info_list: list[onnx.ValueInfoProto],
            model_initalizer_list: list
    ) -> dict:
        tensor_dict = {}
        weight_dict = {}

        for tensor_list in [model_input_list, model_output_list, model_value_info_list]:
            for tensor in tensor_list:    
                shape = ()

                for dim in tensor.type.tensor_type.shape.dim:
                    shape += (dim.dim_value,)
                
                tensor_dict[tensor.name] = {'shape': shape, 'size': DTYPES[tensor_dtype_to_string(tensor.type.tensor_type.elem_type)]}

        for initializer in model_initalizer_list:
            weight_dict[initializer.name] = {'shape': tuple(initializer.dims), 'size': DTYPES[tensor_dtype_to_string(initializer.data_type)]}

        
        return tensor_dict, weight_dict
    

    def getMemoryInfo(
            self,
            node_param_dict: dict
    ) -> tuple[dict, dict]:
        params_dict = {}
        memory_dict = {}

        for node in node_param_dict:
            name, shape, size = node_param_dict[node]
            count = len(name)

            params_size = ()
            memory_size = ()

            for i in range(count):
                param_size = numpy.prod(shape[i], dtype=numpy.int64)

                params_size += ((param_size.item(),),)
                memory_size += (((param_size * size[i]).item(),),)

            params_dict[node] = params_size if params_size else ((0,),)
            memory_dict[node] = memory_size if memory_size else ((0,),)
        
        return params_dict, memory_dict
    

    def profileModel(
            self,
            onnx_model_path: str
    ):
        sess_options = onnxruntime.SessionOptions()

        sess_options.intra_op_num_threads = 1
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")

        sess_options.enable_profiling = True

        ort_session = onnxruntime.InferenceSession(
                onnx_model_path,
                providers=["CPUExecutionProvider"],
                sess_options=sess_options
            )
        
        # model inputs and outputs
        output_feed = [x.name for x in ort_session.get_outputs()]

        input_feed = {}

        for model_input in ort_session.get_inputs():
            input_feed[model_input.name] = get_random_input(model_input.shape, model_input.type)

        ort_session.run(output_feed, input_feed)
        
        prof_file = ort_session.end_profiling()

        prof_path = os.path.join(self.prof_directory, prof_file)

        shutil.move(os.path.join(self.workspace, prof_file), prof_path)

        return prof_path


    def profileMemory(
            self,
            onnx_model_path: str
    ) -> int:
        # onnx model graph for profiling
        graph = onnx.load(onnx_model_path).graph

        # list of onnx.NodeProto
        self.nodes = graph.node

        # list of valid onnx.NodeProto for analysis
        self.valid_nodes_list = [(node.name, node.op_type) for node in self.nodes]

        # model inputs and outputs
        self.inputs = graph.input
        self.outputs = graph.output

        # list of onnx.TensorProto
        self.model_weights = graph.initializer

        # operator and tensor dicts
        self.unsorted_count_operators = {}

        # tensor dict and weights-bias dict
        self.tensor_dict, self.wb_dict = self.updateTensorandWeightDict(self.inputs, self.outputs, graph.value_info, self.model_weights)

        for i, node in enumerate(self.nodes):
            node_input_list = []
            node_wb_list = []
            node_output_list = []

            input_shapes = ()
            input_size = ()

            wb_shapes = ()
            wb_size = ()

            output_shapes = ()
            output_size = ()

            # parse onnx gragh for inputs and weight data
            for i, node_input in enumerate(node.input):
                if node_input:
                    if node_input in self.wb_dict.keys():
                        wb_shape = self.wb_dict[node_input]['shape']

                        if wb_shape:
                            wb_shapes += (wb_shape,)
                            wb_size += (self.wb_dict[node_input]['size'],)

                            node_wb_list.append(node_input)
                    
                    else:
                        input_shape = self.tensor_dict[node_input]['shape']

                        if input_shape:
                            input_shapes += (input_shape,)
                            input_size += (self.tensor_dict[node_input]['size'],)

                            node_input_list.append(node_input)

            self.node_input_dict[node.name] = (tuple(node_input_list), input_shapes, input_size)
            
            self.node_wb_dict[node.name] = (tuple(node_wb_list), wb_shapes, wb_size)
            
            # parse onnx gragh for outputs data
            for i, node_output in enumerate(node.output):
                output_shape = self.tensor_dict[node_output]['shape']

                if output_shape:
                    output_shapes += (output_shape,)
                    output_size += (self.tensor_dict[node_output]['size'],)

                    node_output_list.append(node_output)
            
            if node_output_list:
                self.node_output_dict[node.name] = (tuple(node_output_list), output_shapes, output_size)

            # is node does not have a valid output shape, node can be ignored
            else:
                self.valid_nodes_list.remove((node.name, node.op_type))

            
            if self.unsorted_count_operators.get(node.op_type, None) is None:
                self.unsorted_count_operators[node.op_type] = 1
            
            else:
                self.unsorted_count_operators[node.op_type] += 1
        

        # sort based on operators, in alphabetical order
        operators = list(self.unsorted_count_operators.keys())
        operators.sort()
        self.count_operators = {op: int(self.unsorted_count_operators[op]) for op in operators}

        # remove operators not in self.valid_nodes_list
        valid_ops = set([elem[1] for elem in self.valid_nodes_list])

        for op in list(self.count_operators):
            if op not in valid_ops:
                del self.count_operators[op]

        # params size and memory size for inputs and outputs
        self.input_params_dict, self.input_memory_dict = self.getMemoryInfo(self.node_input_dict)

        self.wb_params_dict, self.wb_memory_dict = self.getMemoryInfo(self.node_wb_dict)

        self.output_params_dict, self.output_memory_dict = self.getMemoryInfo(self.node_output_dict)

        # summarize
        self.summarize()

        return 0
    

    def summarize(
            self
    ) -> None:
        dataframe = pandas.DataFrame(columns=['Node', 'Operator', 'Number of Params', 'Inputs Shape', 'Output Shape',
                                              'Weights and Bias Shape', 'Inputs Size', 'Output Size',
                                              'Inputs Memory (in Bytes)', 'Weights and Bias Memory (in Bytes)',
                                              'Output Memory (in Bytes)'])

        for node, op_type in self.valid_nodes_list:
            row = pandas.DataFrame([[node, op_type,
                                     _convert_shape_tuple_to_string(self.wb_params_dict[node], add=True),
                                     _convert_shape_tuple_to_string(self.node_input_dict[node][1]),
                                     _convert_shape_tuple_to_string(self.node_output_dict[node][1]),
                                     _convert_shape_tuple_to_string(self.node_wb_dict[node][1]),
                                     _convert_shape_tuple_to_string(self.input_params_dict[node], add=True),
                                     _convert_shape_tuple_to_string(self.output_params_dict[node], add=True),
                                     _convert_shape_tuple_to_string(self.input_memory_dict[node], add=True),
                                     _convert_shape_tuple_to_string(self.wb_memory_dict[node], add=True),
                                     _convert_shape_tuple_to_string(self.output_memory_dict[node], add=True)]],
                                     columns=dataframe.columns)
            
            dataframe = pandas.concat([dataframe, row], ignore_index=True)
        
        # type-cast
        dataframe['Number of Params'] = dataframe['Number of Params'].astype('int64')
        dataframe['Inputs Memory (in Bytes)'] = dataframe['Inputs Memory (in Bytes)'].astype('int64')
        dataframe['Weights and Bias Memory (in Bytes)'] = dataframe['Weights and Bias Memory (in Bytes)'].astype('int64')
        dataframe['Output Memory (in Bytes)'] = dataframe['Output Memory (in Bytes)'].astype('int64')
        
        # params percent
        dataframe.insert(3, 'Params (%)', ((dataframe['Number of Params'] * 100.0).astype('float64') / dataframe['Number of Params'].sum()).astype('float64').round(3))

        # memory
        dataframe.insert(4, 'Memory (in Bytes)', (dataframe['Weights and Bias Memory (in Bytes)'] + dataframe['Output Memory (in Bytes)']).astype('int64'))

        # memory percent
        dataframe.insert(5, 'Memory (%)', ((dataframe['Memory (in Bytes)'] * 100.0).astype('float64') / dataframe['Memory (in Bytes)'].sum()).astype('float64').round(3))

        dataframe.insert(6, 'Weights and Bias Memory (%)', ((dataframe['Weights and Bias Memory (in Bytes)'] * 100.0).astype('float64') / 
                                                   (dataframe['Weights and Bias Memory (in Bytes)'].sum() + 
                                                   dataframe['Output Memory (in Bytes)'].sum())).astype('float64').round(3))
        
        dataframe.insert(7, 'Output Memory (%)', ((dataframe['Output Memory (in Bytes)'] * 100.0).astype('float64') / 
                                                   (dataframe['Weights and Bias Memory (in Bytes)'].sum() + 
                                                   dataframe['Output Memory (in Bytes)'].sum())).astype('float64').round(3))
        
        # Memory in MB
        dataframe['Weights and Bias Memory (in MB)'] = (dataframe['Weights and Bias Memory (in Bytes)'] / 1e6).astype('float64').round(6)
        dataframe['Output Memory (in MB)'] = (dataframe['Output Memory (in Bytes)'] / 1e6).astype('float64').round(6)
        dataframe['Memory (in MB)'] = (dataframe['Weights and Bias Memory (in MB)'] + dataframe['Output Memory (in MB)']).astype('float64').round(6)

        # total
        dataframe.loc['Total'] = dataframe.sum(numeric_only=True).round(0)

        dataframe.to_csv(os.path.join(self.profile_logs_directory, self.model_name + '_summary.csv'), index=False, mode='w')

        # grouping by operator
        grouped_dataframe = dataframe[['Operator', 'Number of Params', 'Params (%)', 'Memory (in Bytes)', 'Memory (%)',
                                       'Weights and Bias Memory (in Bytes)', 'Weights and Bias Memory (%)',
                                       'Output Memory (in Bytes)', 'Output Memory (%)']].groupby(['Operator'], as_index=False).sum()

        # operator count and percent
        grouped_dataframe.insert(1, 'Count', list(self.count_operators.values()))
        grouped_dataframe.insert(2, 'Count (%)', ((grouped_dataframe['Count'] * 100.0).astype('float64') / grouped_dataframe['Count'].sum()).astype('float64').round(3))

        # total
        grouped_dataframe.loc['Total'] = grouped_dataframe.sum(numeric_only=True).round(0)

        grouped_dataframe = grouped_dataframe.round(3)

        grouped_dataframe.to_csv(os.path.join(self.profile_logs_directory, self.model_name + '_grouped_summary.csv'), index=False, mode='w')

        logging.debug('Use this command "{}" to view the profiling summary in PowerShell on Windows'.format(self.render(self.profile_logs_directory, self.model_name + '_summary.csv')))


    def modifyGraph(self, delete_block: list, upper_2_ok: bool = False, only_middle: bool = False):
        self.onnx_graph_orig = copy.deepcopy(self.onnx_model.graph)
        self.onnx_graph = self.onnx_model.graph
        
        self.initializers = self.onnx_graph.initializer
        self.initializer_dict = {}
        
        for initializer in self.initializers:
            self.initializer_dict[initializer.name] = initializer
        
        self.nodes = self.onnx_graph_orig.node
        
        self.ouputs = self.onnx_graph.output

        n = 1
        i = 1
        
        # remove nodes
        while n < len(self.nodes) - 1:
            self.prev_node = self.onnx_graph_orig.node[n-1]
            self.current_node = self.onnx_graph_orig.node[n]
            self.next_node = self.onnx_graph_orig.node[n+1]
            
            # prev prev node -> prev node -> current node -> next node -> next next node => prev prev node -> next next node
            # boundary check: i must be equal or greater than 2
            if self.prev_node.op_type == delete_block[0] and self.current_node.op_type == delete_block[1] and self.next_node.op_type == delete_block[2]:
                _prev_node = self.onnx_graph.node[i-1]
                _current_node = self.onnx_graph.node[i]
                _next_node = self.onnx_graph.node[i+1]
                
                _outputs = self.onnx_graph.node[i-2].output
                
                self.onnx_graph.node.remove(_prev_node)
                self.onnx_graph.node.remove(_next_node)
                self.onnx_graph.node.remove(_current_node)
                
                i -= 2
                n += 1
                
                self.onnx_graph.node[i+1].input[0] = _outputs[0]
            
            # prev prev node -> prev node -> current node -> next node => prev prev node -> next node
            elif upper_2_ok and self.prev_node.op_type == delete_block[0] and self.current_node.op_type == delete_block[1] and self.next_node.op_type != delete_block[2]:
                _prev_node = self.onnx_graph.node[i-1]
                _current_node = self.onnx_graph.node[i]
                
                _outputs = self.onnx_graph.node[i-2].output
            
                self.onnx_graph.node.remove(_prev_node)
                self.onnx_graph.node.remove(_current_node)
                
                i -= 2
                
                self.onnx_graph.node[i+1].input[0] = _outputs[0]
            
            elif only_middle and self.current_node.op_type == delete_block[1]:
                _current_node = self.onnx_graph.node[i]
                
                _outputs = self.onnx_graph.node[i-1].output

                self.onnx_graph.node.remove(_current_node)
                
                i -= 1
                
                self.onnx_graph.node[i+1].input[0] = _outputs[0]
        
            i += 1
            n += 1
        
        # remove intializers for the deleted nodes
        remaining_inputs = []
        
        for remaining_nodes in self.onnx_graph.node:
            remaining_inputs += remaining_nodes.input
            
        for initializer_name in self.initializer_dict:
            if initializer_name not in remaining_inputs:
                self.initializers.remove(self.initializer_dict[initializer_name])
        
        try:
            onnx.checker.check_model(self.onnx_model, check_custom_domain=True)
        
        except onnx.checker.ValidationError as e:
            raise Exception(e)
        
        onnx.save_model(self.onnx_model, self.model_name + '_modified.onnx')

