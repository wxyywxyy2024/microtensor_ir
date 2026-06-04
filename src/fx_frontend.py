# src/fx_frontend.py
import torch
import torch.fx
from .frontend import Graph

def import_torch_model(model: torch.nn.Module, example_inputs: tuple) -> Graph:
    """Traces a PyTorch model and lowers it to our custom Graph IR."""
    # 1. Symbolically trace the model using torch.fx
    traced = torch.fx.symbolic_trace(model)
    fx_graph = traced.graph
    
    compiler_graph = Graph()
    # Map to bind torch.fx nodes to our custom TensorValues
    node_map = {}

    # 2. Iterate through the PyTorch FX nodes linear stream
    for fx_node in fx_graph.nodes:
        if fx_node.op == "placeholder":
            # This is an input to the model
            # fx_node.type might be empty, so we infer shape from our example inputs
            idx = len(compiler_graph.inputs)
            shape = tuple(example_inputs[idx].shape)
            
            custom_tensor = compiler_graph.add_input(shape)
            node_map[fx_node] = custom_tensor
            
        elif fx_node.op == "call_function":
            # Identify the operation type
            op_name = fx_node.target.__name__
            
            # Map standard PyTorch functions to our IR namespaces
            if op_name in ["mm", "matmul"]:
                op_type = "MatMul"
            elif op_name in ["add"]:
                op_type = "Add"
            elif op_name in ["relu", "relu_"]:
                op_type = "ReLU"
            else:
                raise NotImplementedError(f"PyTorch operation '{op_name}' is unsupported by µTIR")

            # Extract inputs from the map
            compiler_inputs = [node_map[arg] for arg in fx_node.args if arg in node_map]
            
            # Create the operation in our compiler graph (which automatically triggers shape inference!)
            custom_output = compiler_graph.create_op(op_type, compiler_inputs)
            node_map[fx_node] = custom_output
            
        elif fx_node.op == "output":
            # Link the final graph outputs
            final_args = fx_node.args[0]
            if isinstance(final_args, (tuple, list)):
                for arg in final_args:
                    if arg in node_map:
                        compiler_graph.outputs.append(node_map[arg])
            elif final_args in node_map:
                compiler_graph.outputs.append(node_map[final_args])

    return compiler_graph