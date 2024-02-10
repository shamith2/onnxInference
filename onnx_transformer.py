# This script contains functions for modifying ONNX graphs

import copy
import os
import onnx

class ONNXTransformer:
    def __init__(self, onnx_model_path: str):
        self.model_name = onnx_model_path[:-5]
        
        # load onnx model from ssd
        self.onnx_model = onnx.load(onnx_model_path)
        
        onnx.checker.check_model(self.onnx_model)
            
    def modifyGraph(self, delete_block: (list or tuple), upper_2_ok: bool = False, only_middle: bool = False):
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
            onnx.checker.check_model(self.onnx_model)
        
        except onnx.checker.ValidationError as e:
            raise Exception(e)
        
        onnx.save_model(self.onnx_model, self.model_name + '_modified.onnx')

if __name__ == '__main__':
    model_path = os.path.join(os.path.expanduser('~'), 'IPU', 'onnxruntime', 'onnx_models', 'ryzen_ai', 'mobilenetv2_qdq', 'mobilenetv2_qdq_infer_int8.onnx')
    
    onnx_t = ONNXTransformer(model_path)
    
    onnx_t.modifyGraph(delete_block=['DequantizeLinear', 'Clip', 'QuantizeLinear'], upper_2_ok=False, only_middle=True)