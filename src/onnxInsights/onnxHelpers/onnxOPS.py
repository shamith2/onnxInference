# MACS for ONNX Operators
# Operators as listed on https://onnx.ai/onnx/operators/
# Needs verification

# MACS for atomic-like operators
MEM_MAC = 0
ADD_MAC = 1
MUL_MAC = 1
MUL_ADD_MAC = 1
CMP_MAC = 1
DIV_MAC = 4
SQRT_MAC = 4
LOG_MAC = 4
EXP_MAC = 8
TRIG_MAC = 8

# dict of ONNX Operators
OPERATORS = {
    'OPERATOR': [{'MACS': 0}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 0}],
    'ADD': [{'MACS': ADD_MAC}, {'ADD': 1, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 0}],
    'AND': [{'MACS': CMP_MAC}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 1, 'TRIG': 0, 'MEM': 0}],
    'BATCHNORMALIZATION': [{'MACS': 2*ADD_MAC + MUL_MAC + DIV_MAC + SQRT_MAC}, {'ADD': 2, 'MUL': 1, 'DIV': 1, 'EXP': 0, 'SQRT': 1, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 0}],
    'CAST': [{'MACS': CMP_MAC}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 1, 'TRIG': 0, 'MEM': 0}],
    'CONCAT': [{'MACS': MEM_MAC}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 1}],
    'CONSTANT': [{'MACS': MEM_MAC}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 1}],
    'CONSTANTOFSHAPE': [{'MACS': MEM_MAC}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 1}],
    'COS': [{'MACS': TRIG_MAC}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 1, 'MEM': 0}],
    'DIV': [{'MACS': DIV_MAC}, {'ADD': 0, 'MUL': 0, 'DIV': 1, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 0}],
    'EQUAL': [{'MACS': CMP_MAC}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 1, 'TRIG': 0, 'MEM': 0}],
    'EXPAND': [{'MACS': MUL_MAC}, {'ADD': 0, 'MUL': 1, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 0}],
    'GATHER': [{'MACS': MEM_MAC}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 1}],
    'GREATER': [{'MACS': CMP_MAC}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 1, 'TRIG': 0, 'MEM': 0}],
    'GEMM': [{'MACS': MUL_ADD_MAC}, {'ADD': 1, 'MUL': 1, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 0}],
    'MATMUL': [{'MACS': MUL_MAC}, {'ADD': 0, 'MUL': 1, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 0}],
    'MATMULNBITS': [{'MACS': MUL_MAC}, {'ADD': 0, 'MUL': 1, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 0}],
    'MUL': [{'MACS': MUL_ADD_MAC}, {'ADD': 1, 'MUL': 1, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 0}],
    'NEG': [{'MACS': MUL_MAC}, {'ADD': 0, 'MUL': 1, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 0}],
    'POW': [{'MACS': EXP_MAC}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 1, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 0}],
    'RANGE': [{'MACS': ADD_MAC + 2*CMP_MAC + 1*DIV_MAC}, {'ADD': 1, 'MUL': 0, 'DIV': 1, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 2, 'TRIG': 0, 'MEM': 0}],
    'REDUCEMEAN': [{'MACS': ADD_MAC}, {'ADD': 1, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 0}],
    'RESHAPE': [{'MACS': MEM_MAC}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 1}],
    'SCATTERND': [{'MACS': MEM_MAC}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 1}],
    'SHAPE': [{'MACS': MEM_MAC}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 1}],
    'SIGMOID': [{'MACS': ADD_MAC + MUL_MAC + EXP_MAC + DIV_MAC}, {'ADD': 1, 'MUL': 1, 'DIV': 1, 'EXP': 1, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 0}],
    'SIN': [{'MACS': TRIG_MAC}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 1, 'MEM': 0}],
    'SLICE': [{'MACS': MEM_MAC}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 1}],
    'SOFTMAX': [{'MACS': ADD_MAC + EXP_MAC}, {'ADD': 1, 'MUL': 0, 'DIV': 0, 'EXP': 1, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 0}],
    'SQRT': [{'MACS': SQRT_MAC}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 1, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 0}],
    'TRANSPOSE': [{'MACS': MEM_MAC}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 1}],
    'TRILU': [{'MACS': CMP_MAC}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 1, 'TRIG': 0, 'MEM': 0}],
    'UNSQUEEZE': [{'MACS': MEM_MAC}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0, 'MEM': 1}],
    'WHERE': [{'MACS': CMP_MAC}, {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 1, 'TRIG': 0, 'MEM': 0, 'MEM': 0}],
}

