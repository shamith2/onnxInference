# Number of compute operations for ONNX Operators
# Operators as listed on https://onnx.ai/onnx/operators/
# Needs verification

# OPERATIONS for atomic-like operators
MEM_MACS = 0

# Matrix Engine
ADD_MACS = 1
MUL_MACS = 1
MUL_ADD_MACS = 1

# Vector Engine
CMP_OPS = 1
DIV_FLOPS = 1
SQRT_FLOPS = 1
LOG_FLOPS = 1
EXP_FLOPS = 1
TRIG_FLOPS = 1

# dict of ONNX Operators
OPERATORS = {
    'OPERATOR': [{'OPS': 0}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 0}],
    'ADD': [{'OPS': ADD_MACS}, {'ADD': 1, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 0}],
    'AND': [{'OPS': CMP_OPS}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 1, 'TRIG': 0, 'MEM': 0}],
    'BATCHNORMALIZATION': [{'OPS': ADD_MACS + MUL_ADD_MACS + DIV_FLOPS + SQRT_FLOPS}, {'ADD': 2, 'MUL': 1, 'DIV': 1, 'EXP': 0, 'SQRT': 1, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 0}],
    'CAST': [{'OPS': CMP_OPS}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 1, 'TRIG': 0, 'MEM': 0}],
    'CONCAT': [{'OPS': MEM_MACS}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 1}],
    'CONSTANT': [{'OPS': MEM_MACS}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 1}],
    'CONSTANTOFSHAPE': [{'OPS': MEM_MACS}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 1}],
    'COS': [{'OPS': TRIG_FLOPS}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 1, 'MEM': 0}],
    'DIV': [{'OPS': DIV_FLOPS}, {'ADD': 0, 'MUL': 0, 'DIV': 1, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 0}],
    'EQUAL': [{'OPS': CMP_OPS}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 1, 'TRIG': 0, 'MEM': 0}],
    'EXPAND': [{'OPS': MUL_MACS}, {'ADD': 0, 'MUL': 1, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 0}],
    'GATHER': [{'OPS': MEM_MACS}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 1}],
    'GREATER': [{'OPS': CMP_OPS}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 1, 'TRIG': 0, 'MEM': 0}],
    'GEMM': [{'OPS': MUL_ADD_MACS}, {'ADD': 1, 'MUL': 1, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 0}],
    'MATMUL': [{'OPS': MUL_MACS}, {'ADD': 0, 'MUL': 1, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 0}],
    'MATMULNBITS': [{'OPS': MUL_MACS}, {'ADD': 0, 'MUL': 1, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 0}],
    'MUL': [{'OPS': MUL_ADD_MACS}, {'ADD': 1, 'MUL': 1, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 0}],
    'NEG': [{'OPS': MUL_MACS}, {'ADD': 0, 'MUL': 1, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 0}],
    'POW': [{'OPS': EXP_FLOPS}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 1, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 0}],
    'RANGE': [{'OPS': ADD_MACS + 2*CMP_OPS}, {'ADD': 1, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 2, 'TRIG': 0, 'MEM': 0}],
    'REDUCEMEAN': [{'OPS': ADD_MACS}, {'ADD': 1, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 0}],
    'RESHAPE': [{'OPS': MEM_MACS}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 1}],
    'SCATTERND': [{'OPS': MEM_MACS}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 1}],
    'SHAPE': [{'OPS': MEM_MACS}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 1}],
    'SIGMOID': [{'OPS': MUL_ADD_MACS + EXP_FLOPS + DIV_FLOPS}, {'ADD': 1, 'MUL': 1, 'DIV': 1, 'EXP': 1, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 0}],
    'SIN': [{'OPS': TRIG_FLOPS}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 1, 'MEM': 0}],
    'SLICE': [{'OPS': MEM_MACS}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 1}],
    'SOFTMAX': [{'OPS': ADD_MACS + 2*EXP_FLOPS}, {'ADD': 1, 'MUL': 0, 'DIV': 0, 'EXP': 2, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 0}],
    'SQRT': [{'OPS': SQRT_FLOPS}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 1, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 0}],
    'TRANSPOSE': [{'OPS': MEM_MACS}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 1}],
    'TRILU': [{'OPS': CMP_OPS}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 1, 'TRIG': 0, 'MEM': 0}],
    'UNSQUEEZE': [{'OPS': MEM_MACS}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 1}],
    'WHERE': [{'OPS': CMP_OPS}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 1, 'TRIG': 0, 'MEM': 0, 'MEM': 0}],
}

