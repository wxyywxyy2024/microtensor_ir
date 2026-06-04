# test_run.py
from src.frontend import Graph

g = Graph()

# Define weights and features
X = g.add_input((128, 512))  # Batch size 128, Input features 512
W = g.add_input((512, 256))  # Input features 512, Output features 256
B = g.add_input((128, 256))  # Bias tensor (128, 256)

# Construct computational topology
matmul_out = g.create_op("MatMul", [X, W])
add_out = g.create_op("Add", [matmul_out, B])
final_out = g.create_op("ReLU", [add_out])

g.outputs.append(final_out)

# Print out verified Graph HLIR
g.print_ir()