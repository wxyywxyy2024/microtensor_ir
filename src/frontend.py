from .ir import TensorValue, OpNode

class Graph:
    def __init__(self):
        self.nodes = []
        self.inputs = []
        self.outputs = []
        self._var_counter = 0

    def get_unique_name(self) -> str:
        self._var_counter += 1
        return f"t{self._var_counter}"

    def add_input(self, shape: tuple, dtype: str = "float32") -> TensorValue:
        t = TensorValue(self.get_unique_name(), shape, dtype)
        self.inputs.append(t)
        return t

    def create_op(self, op_type: str, inputs: list) -> TensorValue:
        node_name = f"{op_type.lower()}_{self._var_counter}"
        node = OpNode(op_type, inputs, node_name)
        
        # Shape Inference Validation Engine
        if op_type == "MatMul":
            # Rule: Matrix Multiplication [M, K] x [K, N] -> [M, N]
            assert len(inputs) == 2, "MatMul requires exactly 2 inputs"
            shape_a = inputs[0].shape
            shape_b = inputs[1].shape
            assert len(shape_a) == 2 and len(shape_b) == 2, "Only 2D MatMul supported for now"
            assert shape_a[1] == shape_b[0], f"Dimension mismatch: inner dimensions {shape_a[1]} and {shape_b[0]} must match."
            
            out_shape = (shape_a[0], shape_b[1])
            
        elif op_type == "ReLU" or op_type == "Add":
            # Elementwise verification rules
            if op_type == "Add":
                assert inputs[0].shape == inputs[1].shape, f"Elementwise Add dimensions must match identically: {inputs[0].shape} vs {inputs[1].shape}"
            out_shape = inputs[0].shape
            
        else:
            raise NotImplementedError(f"Unsupported operation: {op_type}")

        # Bind output values to the operator and track the graph topology
        out_tensor = TensorValue(self.get_unique_name(), out_shape, inputs[0].dtype)
        node.outputs.append(out_tensor)
        self.nodes.append(node)
        return out_tensor

    def print_ir(self):
        print("// High-Level Graph IR")
        for inp in self.inputs:
            print(f"Input: {inp}")
        for node in self.nodes:
            print(f"  {node}")