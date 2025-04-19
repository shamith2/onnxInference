import onnx
from onnx import helper, TensorProto
import numpy

# Model parameters
batch_size = 2
input_dim = 256
output_dim = 128

# Create input tensors
input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, (batch_size, input_dim))
weight_tensor = helper.make_tensor_value_info("W", TensorProto.FLOAT, (input_dim, output_dim))
bias_tensor = helper.make_tensor_value_info("B", TensorProto.FLOAT, (batch_size, output_dim))
output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, (batch_size, output_dim))

generator = numpy.random.default_rng(seed=42)

# Create initializers (weights and bias)
W = generator.random(size=(input_dim, output_dim))
B = generator.random(size=(batch_size, output_dim))

W_initializer = helper.make_tensor("W", TensorProto.FLOAT, (input_dim, output_dim), W)
B_initializer = helper.make_tensor("B", TensorProto.FLOAT, (batch_size, output_dim), B)

# GEMM node: Y = alpha * X * W + beta * B
# default: alpha = 1.0, beta = 1.0
gemm_node = helper.make_node(
    "Gemm",
    inputs=("X", "W", "B"),
    outputs=("Y",)
)

# Create the graph
graph_def = helper.make_graph(
    [gemm_node],
    "SimpleGemm",
    inputs=(input_tensor,),
    outputs=(output_tensor,),
    initializer=[W_initializer, B_initializer]
)

# Create the model
model = helper.make_model(graph_def)
onnx.checker.check_model(model, full_check=True)

# Save the model
onnx.save(model, "simple_gemm.onnx")
print("Model saved as simple_gemm.onnx")
