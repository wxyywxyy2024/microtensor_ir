# src/ir.py
from typing import List, Tuple

class TensorValue:
    def __init__(self, name: str, shape: Tuple[int, ...], dtype: str = "float32"):
        self.name = name
        self.shape = shape
        self.dtype = dtype

    def __repr__(self):
        return f"%{self.name} : tensor<{', '.join(map(str, self.shape))}, {self.dtype}>"

class OpNode:
    def __init__(self, op_type: str, inputs: List[TensorValue], name: str):
        self.op_type = op_type      # e.g., 'MatMul', 'ReLU', 'Add'
        self.inputs = inputs        # Inputs are existing TensorValues
        self.name = name            # Unique identifier for the operation node
        self.outputs: List[TensorValue] = []

    def __repr__(self):
        inputs_str = ", ".join([f"%{t.name}" for t in self.inputs])
        outputs_str = ", ".join([str(t) for t in self.outputs])
        return f"{outputs_str} = {self.op_type}({inputs_str})"