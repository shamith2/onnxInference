# MACS for ONNX Operators
# Operators as listed on https://onnx.ai/onnx/operators/
# Needs verification

OPERATORS = {
    'OPERATOR': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0},
    'ABS': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 1, 'TRIG': 0},
    'ACOS': {'ADD': 1, 'MUL': 2, 'DIV': 1, 'EXP': 0, 'SQRT': 1, 'LOG': 0, 'CMP': 0, 'TRIG': 0},
    'ACOSH': {'ADD': 2, 'MUL': 1, 'DIV': 0, 'EXP': 0, 'SQRT': 1, 'LOG': 1, 'CMP': 0, 'TRIG': 0},
    'ADD': {'ADD': 1, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0},
    'AFFINEGRID': {'ADD': 1, 'MUL': 1, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0},
    'AND': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 1, 'TRIG': 0},
    'ARGMAX': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 1, 'TRIG': 0},
    'ARGMIN': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 1, 'TRIG': 0},
    'ASIN': {'ADD': 1, 'MUL': 2, 'DIV': 1, 'EXP': 0, 'SQRT': 1, 'LOG': 0, 'CMP': 0, 'TRIG': 0},
    'ASINH': {'ADD': 2, 'MUL': 1, 'DIV': 0, 'EXP': 0, 'SQRT': 1, 'LOG': 1, 'CMP': 0, 'TRIG': 0},
    'ATAN': {'ADD': 2, 'MUL': 2, 'DIV': 1, 'EXP': 0, 'SQRT': 1, 'LOG': 0, 'CMP': 0, 'TRIG': 0},
    'ATANH': {'ADD': 2, 'MUL': 1, 'DIV': 1, 'EXP': 0, 'SQRT': 0, 'LOG': 1, 'CMP': 0, 'TRIG': 0},
    'AVERAGEPOOL': {'ADD': 1, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0},
    'BATCHNORMALIZATION': {'ADD': 2, 'MUL': 1, 'DIV': 1, 'EXP': 0, 'SQRT': 1, 'LOG': 0, 'CMP': 0, 'TRIG': 0},
    'BERNOULLI': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0}, # TODO
    'BITSHIFT': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0}, # TODO
    'BITWISEAND': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 1, 'TRIG': 0},
    'BITWISENOT': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 1, 'TRIG': 0},
    'BITWISEOR': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 1, 'TRIG': 0},
    'BITWISEXOR': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 1, 'TRIG': 0},
    'BLACKMANWINDOW': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0}, # TODO
    'CAST': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 1, 'TRIG': 0},
    'CASTLIKE': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 1, 'TRIG': 0},
    'CEIL': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 1, 'TRIG': 0},
    'CELU': {'ADD': 1, 'MUL': 1, 'DIV': 0, 'EXP': 1, 'SQRT': 0, 'LOG': 0, 'CMP': 2, 'TRIG': 0},
    'CENTERCROPPAD': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0}, # TODO
    'CLIP': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 2, 'TRIG': 0},
    'COL2IM': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0},
    'COMPRESS': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 1, 'TRIG': 0},
    'CONCAT': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0},
    'CONCATFROMSEQUENCE': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0},
    'CONSTANT': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0},
    'CONSTANTOFSHAPE': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0},
    'CONV': {'ADD': 1, 'MUL': 1, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0},
    'CONVINTEGER': {'ADD': 1, 'MUL': 1, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0},
    'CONVINTEGER': {'ADD': 1, 'MUL': 1, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0},
    'CONVTRANSPOSE': {'ADD': 1, 'MUL': 1, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0},
    'COS': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 1},
    'COSH': {'ADD': 1, 'MUL': 1, 'DIV': 0, 'EXP': 2, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0},
    'CUMSUM': {'ADD': 1, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0},
    'DFT': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0}, # TODO
    'DEFORMCONV': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0}, # TODO
    'DEPTHTOSPACE': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0},
    'DEQUANTIZELINEAR': {'ADD': 1, 'MUL': 1, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0},
    'DET': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0}, # TODO
    'DIV': {'ADD': 0, 'MUL': 0, 'DIV': 1, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0},
    'DROPOUT': {'ADD': 0, 'MUL': 2, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 1, 'TRIG': 0},
    'DYNAMICQUANTIZELINEAR': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0}, # TODO
    'EINSUM': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0},
    'ELU': {'ADD': 1, 'MUL': 1, 'DIV': 0, 'EXP': 1, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0},
    'EQUAL': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 1, 'TRIG': 0},
    'ERF': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0}, # TODO
    'EXP': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 1, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0},
    'EXPAND': {'ADD': 0, 'MUL': 1, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0},
    'EYELIKE': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 1, 'TRIG': 0},
    'FLATTEN': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0}, # TODO
    'FLOOR': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 1, 'TRIG': 0},
    'GRU': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0}, # TODO
    'GATHER': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0},
    'GATHERELEMENTS': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0}, # TODO
    'GATHERND': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0}, # TODO
    'GELU': {'ADD': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0}, # TODO
    'GEMM': {'ADD': 1, 'MUL': 1, 'DIV': 0, 'EXP': 0, 'SQRT': 0, 'LOG': 0, 'CMP': 0, 'TRIG': 0}, # VERIFY
}