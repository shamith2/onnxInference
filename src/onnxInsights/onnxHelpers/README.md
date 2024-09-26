### Using ONNXProfiler and MemoryView

There are 3 files that comprise the profiler under src/onnxInsights/onnxHelpers:

- onnxProfiler.py: This file contains the code to take the onnx graph, run shape inference (fn shapeInfer) on the graph to get the inputs, weights and output shape of the operators in the graph and generate 2 main csv files:
  - one file that lists the memory and compute info (fn getMemoryandComputeInfo) of each operator in the graph, and
  - the other file tracks if each output of an operator is used as input to operators later in the graph (fn trackOutputs)

- onnxOPS.py: This file lists the different onnx operators with approximate number of compute operations it takes to run each operator (in terms of MAC (multiply-accumulate) and Vector atomic operations). Currently, it only has operators used in Llama3

- MemoryView.py: This file contains the code to parse the generated csv files, see if any operator can be fused when, for example, if the operator uses the output from its previous operator as its input and, given the memory and cache size, determines if an output (of an operator) can be stored in local memory, cache or main memory. Using this info, the total read memory (inputs memory + weights memory) and write memory (outputs memory) from/to the main memory can be plotted

The script to invoke the profiler with LLMs (currently Llama3 and Gemma) is located at scripts/onnxProfile/onnx_llm_profiling.py
